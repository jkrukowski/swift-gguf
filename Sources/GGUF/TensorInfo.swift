import BinaryParsing
import Foundation

extension GGUF {
    public struct TensorInfo: Sendable {
        public let name: String
        public let dimensionCount: UInt32
        public let dimensions: [UInt64]
        public let dataType: GGUF.TensorType
        public let offset: UInt64

        /// Total number of elements in the tensor
        public var elementCount: UInt64 {
            dimensions.reduce(1, *)
        }

        /// Size of this tensor's data in bytes
        public var sizeInBytes: Int {
            dataType.sizeInBytes(elementCount: elementCount)
        }

        public init(
            name: String,
            dimensionCount: UInt32,
            dimensions: [UInt64],
            dataType: GGUF.TensorType,
            offset: UInt64
        ) {
            self.name = name
            self.dimensionCount = dimensionCount
            self.dimensions = dimensions
            self.dataType = dataType
            self.offset = offset
        }
    }
}

extension GGUF.TensorInfo: ExpressibleByParsing {
    public init(parsing input: inout ParserSpan) throws {
        self.name = try String(parsingGGUFString: &input)
        guard name.utf8.count <= GGUF.Constants.maxTensorNameBytes else {
            throw GGUF.Error.invalidTensorName(name)
        }
        self.dimensionCount = try UInt32(parsingLittleEndian: &input)
        guard let dimCount = Int(exactly: dimensionCount) else {
            throw GGUF.Error.invalidTensorDimensionCount(dimensionCount)
        }
        self.dimensions = try Array(parsing: &input, count: dimCount) { span in
            try UInt64(parsingLittleEndian: &span)
        }
        self.dataType = try GGUF.TensorType(parsing: &input)
        self.offset = try UInt64(parsingLittleEndian: &input)
    }
}
