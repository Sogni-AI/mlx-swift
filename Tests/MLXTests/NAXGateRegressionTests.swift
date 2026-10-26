// Correctness guard for the NAX (matrix-unit) matmul paths on generation-17 GPUs.
//
// The owned MLX fork used to gate NAX to GPU generation 18+ because an early
// generation-17 NAX path returned wrong results for fp16 GEMM at M >= 8,
// N >= 8192 and for affine quantized matmul at M >= 64, N >= 9216. The fork now
// follows upstream's rule (generation 17 enabled on Macs), so these shapes are
// checked against a CPU reference: on a machine without NAX they exercise the
// ordinary kernels and must pass as well.

import Foundation
import MLX
import MLXRandom
import XCTest

class NAXGateRegressionTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    private func relativeError(_ actual: MLXArray, _ reference: MLXArray) -> Float {
        let difference = actual - reference
        let differenceFrobenius = (difference * difference).sum().sqrt().item(Float.self)
        let referenceFrobenius = (reference * reference).sum().sqrt().item(Float.self)
        return differenceFrobenius / referenceFrobenius
    }

    func testNAXThresholdFloat16MatmulMatchesCPUReference() {
        MLXRandom.seed(0)
        let m = 16
        let k = 512
        let n = 16384

        // Round the inputs to fp16 first so the comparison isolates accumulation, and
        // compute the reference on the CPU: a GPU float32 matmul may itself run as TF32
        // on NAX hardware and is not a valid reference there.
        let a = MLXRandom.normal([m, k]).asType(.float16)
        let b = MLXRandom.normal([k, n]).asType(.float16)
        eval(a, b)
        let reference = matmul(
            a.asType(.float32, stream: .cpu), b.asType(.float32, stream: .cpu), stream: .cpu)
        eval(reference)

        let actual = matmul(a, b).asType(.float32)
        let error = relativeError(actual, reference)

        XCTAssertLessThan(
            error,
            0.01,
            "fp16 matmul M=\(m) K=\(k) N=\(n) relative error \(error) exceeds 0.01; "
                + "check is_nax_available() and the NAX steel GEMM"
        )
    }

    func testNAXThresholdAffineQuantizedMatmulMatchesCPUReference() {
        MLXRandom.seed(1)
        let m = 64
        let k = 5376
        let n = 9216
        let groupSize = 64
        let bits = 8

        let weight = MLXRandom.normal([n, k], scale: 1 / Float(k).squareRoot()).asType(.bfloat16)
        let packed = quantized(weight, groupSize: groupSize, bits: bits)
        let x = MLXRandom.normal([m, k], scale: 1 / Float(k).squareRoot()).asType(.bfloat16)
        eval(x, packed.wq, packed.scales, packed.biases ?? packed.scales)

        let dequantizedWeight = dequantized(
            packed.wq, scales: packed.scales, biases: packed.biases,
            groupSize: groupSize, bits: bits, dtype: .float32, stream: .cpu
        ).transposed(1, 0, stream: .cpu)
        let reference = matmul(x.asType(.float32, stream: .cpu), dequantizedWeight, stream: .cpu)
        eval(reference)

        let actual = quantizedMM(
            x, packed.wq, scales: packed.scales, biases: packed.biases,
            transpose: true, groupSize: groupSize, bits: bits
        ).asType(.float32)
        let error = relativeError(actual, reference)

        XCTAssertLessThan(
            error,
            0.02,
            "affine-\(bits) g\(groupSize) quantized matmul M=\(m) K=\(k) N=\(n) relative error "
                + "\(error) exceeds 0.02; check is_nax_available() and qmm_nax"
        )
    }
}
