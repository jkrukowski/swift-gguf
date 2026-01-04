import BinaryParsing
import Foundation

extension GGUF {
    public enum TensorType: UInt32, Sendable {
        case f32 = 0
        case f16 = 1
        case q4_0 = 2
        case q4_1 = 3
        case q5_0 = 6
        case q5_1 = 7
        case q8_0 = 8
        case q8_1 = 9
        case q2_K = 10
        case q3_K = 11
        case q4_K = 12
        case q5_K = 13
        case q6_K = 14
        case q8_K = 15
        case iq2_XXS = 16
        case iq2_XS = 17
        case iq3_XXS = 18
        case iq1_S = 19
        case iq4_NL = 20
        case iq3_S = 21
        case iq2_S = 22
        case iq4_XS = 23
        case i8 = 24
        case i16 = 25
        case i32 = 26
        case i64 = 27
        case f64 = 28
        case iq1_M = 29
        case bf16 = 30
        case tq1_0 = 34
        case tq2_0 = 35
        case mxfp4 = 39
    }
}

extension GGUF.TensorType: ExpressibleByParsing {
    public init(parsing input: inout ParserSpan) throws {
        let rawValue = try UInt32(parsingLittleEndian: &input)
        guard let dataType = GGUF.TensorType(rawValue: rawValue) else {
            throw GGUF.Error.invalidTensorType(rawValue)
        }
        self = dataType
    }
}

extension GGUF.TensorType: CustomStringConvertible {
    public var description: String {
        switch self {
        case .f32: "F32"
        case .f16: "F16"
        case .q4_0: "Q4_0"
        case .q4_1: "Q4_1"
        case .q5_0: "Q5_0"
        case .q5_1: "Q5_1"
        case .q8_0: "Q8_0"
        case .q8_1: "Q8_1"
        case .q2_K: "Q2_K"
        case .q3_K: "Q3_K"
        case .q4_K: "Q4_K"
        case .q5_K: "Q5_K"
        case .q6_K: "Q6_K"
        case .q8_K: "Q8_K"
        case .iq2_XXS: "IQ2_XXS"
        case .iq2_XS: "IQ2_XS"
        case .iq3_XXS: "IQ3_XXS"
        case .iq1_S: "IQ1_S"
        case .iq4_NL: "IQ4_NL"
        case .iq3_S: "IQ3_S"
        case .iq2_S: "IQ2_S"
        case .iq4_XS: "IQ4_XS"
        case .i8: "I8"
        case .i16: "I16"
        case .i32: "I32"
        case .i64: "I64"
        case .f64: "F64"
        case .iq1_M: "IQ1_M"
        case .bf16: "BF16"
        case .tq1_0: "TQ1_0"
        case .tq2_0: "TQ2_0"
        case .mxfp4: "MXFP4"
        }
    }

    /// Block size for quantized types (number of elements per block)
    public var blockSize: Int {
        switch self {
        case .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1: 32
        case .q2_K, .q3_K, .q4_K, .q5_K, .q6_K, .q8_K: 256
        case .iq2_XXS, .iq2_XS, .iq3_XXS, .iq1_S, .iq4_NL, .iq3_S, .iq2_S, .iq4_XS, .iq1_M: 256
        case .tq1_0, .tq2_0: 256
        case .mxfp4: 32
        default: 1  // Non-quantized types
        }
    }

    /// Bytes per block for quantized types, or bytes per element for non-quantized types
    public var bytesPerBlock: Int {
        switch self {
        // Non-quantized types (bytes per element)
        case .f32, .i32: 4
        case .f16, .i16, .bf16: 2
        case .i8: 1
        case .f64, .i64: 8

        // Quantized types (bytes per 32-element block)
        case .q4_0: 18  // 32 elements in 18 bytes (2 + 16)
        case .q4_1: 20  // 32 elements in 20 bytes (4 + 16)
        case .q5_0: 22  // 32 elements in 22 bytes (2 + 4 + 16)
        case .q5_1: 24  // 32 elements in 24 bytes (4 + 4 + 16)
        case .q8_0: 34  // 32 elements in 34 bytes (2 + 32)
        case .q8_1: 36  // 32 elements in 36 bytes (4 + 4 + 32)

        // K-quantizations (256 elements per block)
        case .q2_K: 82  // 256 elements in 82 bytes
        case .q3_K: 110  // 256 elements in 110 bytes
        case .q4_K: 144  // 256 elements in 144 bytes
        case .q5_K: 176  // 256 elements in 176 bytes
        case .q6_K: 210  // 256 elements in 210 bytes
        case .q8_K: 292  // 256 elements in 292 bytes

        // IQ quantizations (256 elements per block)
        case .iq2_XXS: 66
        case .iq2_XS: 74
        case .iq3_XXS: 98
        case .iq1_S: 50
        case .iq4_NL: 130
        case .iq3_S: 110
        case .iq2_S: 82
        case .iq4_XS: 136
        case .iq1_M: 56

        // Ternary quantizations
        case .tq1_0: 34
        case .tq2_0: 68

        // Mixed precision
        case .mxfp4: 20
        }
    }

    /// Calculate the total size in bytes for a given number of elements
    public func sizeInBytes(elementCount: UInt64) -> Int {
        let blockSize = self.blockSize
        let bytesPerBlock = self.bytesPerBlock

        if blockSize == 1 {
            // Non-quantized: simple multiplication
            return Int(elementCount) * bytesPerBlock
        } else {
            // Quantized: calculate number of blocks
            let numBlocks = (Int(elementCount) + blockSize - 1) / blockSize
            return numBlocks * bytesPerBlock
        }
    }
}
