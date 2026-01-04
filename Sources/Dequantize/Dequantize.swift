import Foundation
import GGMLDequantize

public enum Dequantize {

    // MARK: - Q4_0

    public static func Q4_0(_ data: Data, elementCount: Int) -> [Float] {
        data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
            [Float](unsafeUninitializedCapacity: elementCount) { outputBuffer, finalCount in
                let inputPtr = ptr.baseAddress?.assumingMemoryBound(to: block_q4_0.self)
                dequantize_row_q4_0(inputPtr, outputBuffer.baseAddress, Int64(elementCount))
                finalCount = elementCount
            }
        }
    }

    // MARK: - Q4_1

    public static func Q4_1(_ data: Data, elementCount: Int) -> [Float] {
        data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
            [Float](unsafeUninitializedCapacity: elementCount) { outputBuffer, finalCount in
                let inputPtr = ptr.baseAddress?.assumingMemoryBound(to: block_q4_1.self)
                dequantize_row_q4_1(inputPtr, outputBuffer.baseAddress, Int64(elementCount))
                finalCount = elementCount
            }
        }
    }

    // MARK: - Q5_0

    public static func Q5_0(_ data: Data, elementCount: Int) -> [Float] {
        data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
            [Float](unsafeUninitializedCapacity: elementCount) { outputBuffer, finalCount in
                let inputPtr = ptr.baseAddress?.assumingMemoryBound(to: block_q5_0.self)
                dequantize_row_q5_0(inputPtr, outputBuffer.baseAddress, Int64(elementCount))
                finalCount = elementCount
            }
        }
    }

    // MARK: - Q5_1

    public static func Q5_1(_ data: Data, elementCount: Int) -> [Float] {
        data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
            [Float](unsafeUninitializedCapacity: elementCount) { outputBuffer, finalCount in
                let inputPtr = ptr.baseAddress?.assumingMemoryBound(to: block_q5_1.self)
                dequantize_row_q5_1(inputPtr, outputBuffer.baseAddress, Int64(elementCount))
                finalCount = elementCount
            }
        }
    }

    // MARK: - Q8_0

    public static func Q8_0(_ data: Data, elementCount: Int) -> [Float] {
        data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
            [Float](unsafeUninitializedCapacity: elementCount) { outputBuffer, finalCount in
                let inputPtr = ptr.baseAddress?.assumingMemoryBound(to: block_q8_0.self)
                dequantize_row_q8_0(inputPtr, outputBuffer.baseAddress, Int64(elementCount))
                finalCount = elementCount
            }
        }
    }

    // MARK: - Q2_K

    public static func Q2_K(_ data: Data, elementCount: Int) -> [Float] {
        data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
            [Float](unsafeUninitializedCapacity: elementCount) { outputBuffer, finalCount in
                let inputPtr = ptr.baseAddress?.assumingMemoryBound(to: block_q2_K.self)
                dequantize_row_q2_K(inputPtr, outputBuffer.baseAddress, Int64(elementCount))
                finalCount = elementCount
            }
        }
    }

    // MARK: - Q3_K

    public static func Q3_K(_ data: Data, elementCount: Int) -> [Float] {
        data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
            [Float](unsafeUninitializedCapacity: elementCount) { outputBuffer, finalCount in
                let inputPtr = ptr.baseAddress?.assumingMemoryBound(to: block_q3_K.self)
                dequantize_row_q3_K(inputPtr, outputBuffer.baseAddress, Int64(elementCount))
                finalCount = elementCount
            }
        }
    }

    // MARK: - Q4_K

    public static func Q4_K(_ data: Data, elementCount: Int) -> [Float] {
        data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
            [Float](unsafeUninitializedCapacity: elementCount) { outputBuffer, finalCount in
                let inputPtr = ptr.baseAddress?.assumingMemoryBound(to: block_q4_K.self)
                dequantize_row_q4_K(inputPtr, outputBuffer.baseAddress, Int64(elementCount))
                finalCount = elementCount
            }
        }
    }

    // MARK: - Q5_K

    public static func Q5_K(_ data: Data, elementCount: Int) -> [Float] {
        data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
            [Float](unsafeUninitializedCapacity: elementCount) { outputBuffer, finalCount in
                let inputPtr = ptr.baseAddress?.assumingMemoryBound(to: block_q5_K.self)
                dequantize_row_q5_K(inputPtr, outputBuffer.baseAddress, Int64(elementCount))
                finalCount = elementCount
            }
        }
    }

    // MARK: - Q6_K

    public static func Q6_K(_ data: Data, elementCount: Int) -> [Float] {
        data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
            [Float](unsafeUninitializedCapacity: elementCount) { outputBuffer, finalCount in
                let inputPtr = ptr.baseAddress?.assumingMemoryBound(to: block_q6_K.self)
                dequantize_row_q6_K(inputPtr, outputBuffer.baseAddress, Int64(elementCount))
                finalCount = elementCount
            }
        }
    }

    // MARK: - Q8_K

    public static func Q8_K(_ data: Data, elementCount: Int) -> [Float] {
        data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
            [Float](unsafeUninitializedCapacity: elementCount) { outputBuffer, finalCount in
                let inputPtr = ptr.baseAddress?.assumingMemoryBound(to: block_q8_K.self)
                dequantize_row_q8_K(inputPtr, outputBuffer.baseAddress, Int64(elementCount))
                finalCount = elementCount
            }
        }
    }
}
