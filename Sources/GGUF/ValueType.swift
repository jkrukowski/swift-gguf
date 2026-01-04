import BinaryParsing
import Foundation

/// GGUF metadata value types
extension GGUF {
    public enum ValueType: UInt32, Sendable, Hashable {
        case uint8 = 0
        case int8 = 1
        case uint16 = 2
        case int16 = 3
        case uint32 = 4
        case int32 = 5
        case float32 = 6
        case bool = 7
        case string = 8
        case array = 9
        case uint64 = 10
        case int64 = 11
        case float64 = 12
    }
}

extension GGUF.ValueType: ExpressibleByParsing {
    public init(parsing input: inout ParserSpan) throws {
        let rawValue = try UInt32(parsingLittleEndian: &input)
        guard let valueType = GGUF.ValueType(rawValue: rawValue) else {
            throw GGUF.Error.invalidValueType(rawValue)
        }
        self = valueType
    }
}
