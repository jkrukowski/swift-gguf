import BinaryParsing

extension GGUF {
    public struct MetadataKeyValue: Sendable {
        public let key: String
        public let value: GGUF.MetadataValue
        public let valueType: GGUF.ValueType

        public init(
            key: String,
            value: GGUF.MetadataValue,
            valueType: GGUF.ValueType
        ) {
            self.key = key
            self.value = value
            self.valueType = valueType
        }
    }
}

extension GGUF.MetadataKeyValue: ExpressibleByParsing {
    public init(parsing input: inout ParserSpan) throws {
        let key = try String(parsingGGUFString: &input)
        let valueType = try GGUF.ValueType(parsing: &input)
        let value = try GGUF.MetadataValue(parsing: &input, type: valueType)
        self.init(
            key: key,
            value: value,
            valueType: valueType
        )
    }
}
