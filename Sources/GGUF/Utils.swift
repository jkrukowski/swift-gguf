import Foundation

extension GGUF {
    public enum Error: Swift.Error {
        case invalidStringCount(UInt64)
        case invalidMetadataCount(UInt64)
        case invalidValueType(UInt32)
        case invalidTensorType(UInt32)
        case invalidTensorName(String)
        case invalidTensorCount(UInt64)
        case invalidTensorDimensionCount(UInt32)
        case invalidArrayLength(UInt64)
        case invalidMagicNumber(UInt32)
        case invalidAlignmentPadding
        case notSupportedVersion(UInt32)
        case unsupportedTensorTypeForConversion(TensorType)
    }
}

extension GGUF {
    enum Constants {
        static let headerMagic: UInt32 = 0x4747_5546
        static let defaultGeneralAlignment = 32
        static let maxTensorNameBytes = 64
    }
}

extension Data {
    func toArray<T>(of type: T.Type, _ fn: (T) -> Float) -> [Float] {
        withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
            let inputBuffer = ptr.bindMemory(to: T.self)
            return inputBuffer.map { fn($0) }
        }
    }
}
