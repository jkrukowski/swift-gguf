import BinaryParsing
import Foundation

/// Metadata value with type-specific data
extension GGUF {
    public enum MetadataValue: Sendable, Hashable {
        case uint8(UInt8)
        case uint16(UInt16)
        case uint32(UInt32)
        case uint64(UInt64)
        case int8(Int8)
        case int16(Int16)
        case int32(Int32)
        case int64(Int64)
        case float32(Float)
        case float64(Double)
        case bool(Bool)
        case string(String)
        case array(GGUF.ValueType, [GGUF.MetadataValue])
    }
}

extension GGUF.MetadataValue {
    init(parsing input: inout ParserSpan, type: GGUF.ValueType) throws {
        switch type {
        case .uint8:
            self = .uint8(try UInt8(parsing: &input))
        case .int8:
            self = .int8(try Int8(parsing: &input))
        case .uint16:
            self = .uint16(try UInt16(parsingLittleEndian: &input))
        case .int16:
            self = .int16(try Int16(parsingLittleEndian: &input))
        case .uint32:
            self = .uint32(try UInt32(parsingLittleEndian: &input))
        case .int32:
            self = .int32(try Int32(parsingLittleEndian: &input))
        case .float32:
            let bits = try UInt32(parsingLittleEndian: &input)
            self = .float32(Float(bitPattern: bits))
        case .bool:
            let value = try UInt8(parsing: &input)
            self = .bool(value != 0)
        case .string:
            self = .string(try String(parsingGGUFString: &input))
        case .array:
            let arrayType = try GGUF.ValueType(parsing: &input)
            let arrayLength = try UInt64(parsingLittleEndian: &input)
            guard let count = Int(exactly: arrayLength) else {
                throw GGUF.Error.invalidArrayLength(arrayLength)
            }
            let values = try Array(parsing: &input, count: count) { span in
                try GGUF.MetadataValue(parsing: &span, type: arrayType)
            }
            self = .array(arrayType, values)
        case .uint64:
            self = .uint64(try UInt64(parsingLittleEndian: &input))
        case .int64:
            self = .int64(try Int64(parsingLittleEndian: &input))
        case .float64:
            let bits = try UInt64(parsingLittleEndian: &input)
            self = .float64(Double(bitPattern: bits))
        }
    }
}

extension GGUF.MetadataValue: CustomStringConvertible {
    public var description: String {
        switch self {
        case .uint8(let v): "\(v) (uint8)"
        case .int8(let v): "\(v) (int8)"
        case .uint16(let v): "\(v) (uint16)"
        case .int16(let v): "\(v) (int16)"
        case .uint32(let v): "\(v) (uint32)"
        case .int32(let v): "\(v) (int32)"
        case .float32(let v): "\(v) (float32)"
        case .bool(let v): "\(v) (bool)"
        case .string(let v): "\"\(v)\""
        case .array(let type, let values): "[\(values.count) x \(type)]"
        case .uint64(let v): "\(v) (uint64)"
        case .int64(let v): "\(v) (int64)"
        case .float64(let v): "\(v) (float64)"
        }
    }
}
