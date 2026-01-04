import BinaryParsing
import Dequantize
import Foundation

public struct GGUF: Sendable {
    public let header: Header
    public let metadata: [MetadataKeyValue]
    public let tensorInfos: [TensorInfo]
    public let tensorNameToIndex: [String: Int]
    public let metadataKeyToValue: [String: MetadataValue]
    public let tensorDataOffset: Int
    public let alignment: Int

    public init(
        header: Header,
        metadata: [MetadataKeyValue],
        tensorInfos: [TensorInfo],
        tensorNameToIndex: [String: Int],
        metadataKeyToValue: [String: MetadataValue],
        tensorDataOffset: Int,
        alignment: Int
    ) {
        self.header = header
        self.metadata = metadata
        self.tensorInfos = tensorInfos
        self.tensorNameToIndex = tensorNameToIndex
        self.metadataKeyToValue = metadataKeyToValue
        self.tensorDataOffset = tensorDataOffset
        self.alignment = alignment
    }

    /// Convenience accessor for metadata by key
    public func metadataValue(forKey key: String) -> GGUF.MetadataValue? {
        metadataKeyToValue[key]
    }

    /// Extract raw tensor data for a specific tensor
    /// - Parameters:
    ///   - tensorIndex: Index of the tensor in tensorInfos array
    ///   - fileData: The complete GGUF file data
    /// - Returns: Raw bytes of the tensor data
    public func tensorData(at tensorIndex: Int, from fileData: Data) -> Data {
        let info = tensorInfos[tensorIndex]
        let startOffset = tensorDataOffset + Int(info.offset)
        let endOffset = startOffset + info.sizeInBytes
        return fileData[startOffset..<endOffset]
    }

    public func tensorData(_ tensorName: String, from fileData: Data) -> Data? {
        guard let tensorIndex = tensorNameToIndex[tensorName] else {
            return nil
        }
        return tensorData(at: tensorIndex, from: fileData)
    }

    /// Extract tensor data as a Float array, dequantizing if necessary
    /// - Parameters:
    ///   - tensorIndex: Index of the tensor in tensorInfos array
    ///   - fileData: The complete GGUF file data
    /// - Returns: Array of Float values
    /// - Throws: Error if the tensor type is not supported for conversion
    public func tensorFloatArray(at tensorIndex: Int, from fileData: Data) throws -> [Float] {
        let info = tensorInfos[tensorIndex]
        let data = tensorData(at: tensorIndex, from: fileData)
        let elementCount = Int(info.elementCount)

        switch info.dataType {
        case .f64:
            return data.toArray(of: Double.self, Float.init)
        case .f32:
            return data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) in
                let inputBuffer = ptr.assumingMemoryBound(to: Float.self)
                return Array(inputBuffer)
            }
        case .f16:
            return data.toArray(of: Float16.self, Float.init)
        case .i8:
            return data.toArray(of: Int8.self, Float.init)
        case .i16:
            return data.toArray(of: Int16.self, Float.init)
        case .i32:
            return data.toArray(of: Int32.self, Float.init)
        case .i64:
            return data.toArray(of: Int64.self, Float.init)
        case .q4_0:
            return Dequantize.Q4_0(data, elementCount: elementCount)
        case .q4_1:
            return Dequantize.Q4_1(data, elementCount: elementCount)
        case .q5_0:
            return Dequantize.Q5_0(data, elementCount: elementCount)
        case .q5_1:
            return Dequantize.Q5_1(data, elementCount: elementCount)
        case .q8_0:
            return Dequantize.Q8_0(data, elementCount: elementCount)
        case .q2_K:
            return Dequantize.Q2_K(data, elementCount: elementCount)
        case .q3_K:
            return Dequantize.Q3_K(data, elementCount: elementCount)
        case .q4_K:
            return Dequantize.Q4_K(data, elementCount: elementCount)
        case .q5_K:
            return Dequantize.Q5_K(data, elementCount: elementCount)
        case .q6_K:
            return Dequantize.Q6_K(data, elementCount: elementCount)
        case .q8_K:
            return Dequantize.Q8_K(data, elementCount: elementCount)
        case .q8_1, .bf16, .iq2_XXS, .iq2_XS, .iq3_XXS, .iq1_S, .iq4_NL, .iq3_S, .iq2_S, .iq4_XS,
            .iq1_M, .tq1_0, .tq2_0, .mxfp4:
            throw Error.unsupportedTensorTypeForConversion(info.dataType)
        }
    }

    public func tensorFloatArray(_ tensorName: String, from fileData: Data) throws -> [Float]? {
        guard let tensorIndex = tensorNameToIndex[tensorName] else {
            return nil
        }
        return try tensorFloatArray(at: tensorIndex, from: fileData)
    }

    /// Extract alignment from metadata (general.alignment key)
    private static func extractAlignment(from metadataKeyToValue: [String: MetadataValue]) -> Int {
        switch metadataKeyToValue["general.alignment"] {
        case .uint32(let alignment):
            return Int(alignment)
        default:
            return Constants.defaultGeneralAlignment
        }
    }
}

extension GGUF: ExpressibleByParsing {
    public init(parsing input: inout ParserSpan) throws {
        let startCount = input.count
        let header = try Header(parsing: &input)
        guard let metadataCount = Int(exactly: header.metadataKeyValueCount) else {
            throw Error.invalidMetadataCount(header.metadataKeyValueCount)
        }
        let metadata = try Array(parsing: &input, count: metadataCount) { span in
            try MetadataKeyValue(parsing: &span)
        }
        guard let tensorCount = Int(exactly: header.tensorCount) else {
            throw Error.invalidTensorCount(header.tensorCount)
        }
        let tensorInfos = try Array(parsing: &input, count: tensorCount) { span in
            try TensorInfo(parsing: &span)
        }
        let metadataKeyToValue = Dictionary(
            uniqueKeysWithValues: metadata.map { ($0.key, $0.value) })
        let alignment = Self.extractAlignment(from: metadataKeyToValue)
        let currentOffset = startCount - input.count
        // Apply alignment padding
        let alignedOffset: Int
        if currentOffset % alignment != 0 {
            let padding = alignment - (currentOffset % alignment)
            // Verify padding bytes are all 0x00
            let paddingBytes = try [UInt8](parsing: &input, count: padding) { span in
                try UInt8(parsing: &span)
            }
            guard paddingBytes.allSatisfy({ $0 == 0x00 }) else {
                throw Error.invalidAlignmentPadding
            }
            alignedOffset = currentOffset + padding
        } else {
            alignedOffset = currentOffset
        }
        self.init(
            header: header,
            metadata: metadata,
            tensorInfos: tensorInfos,
            tensorNameToIndex: Dictionary(
                uniqueKeysWithValues: tensorInfos.enumerated().map { ($0.element.name, $0.offset) }),
            metadataKeyToValue: metadataKeyToValue,
            tensorDataOffset: alignedOffset,
            alignment: alignment
        )
    }
}
