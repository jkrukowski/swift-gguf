import BinaryParsing
import Foundation
import Testing

@testable import GGUF

@Test func `parsing minimal valid data`() throws {
    var data: [UInt8] = []

    // Header: 0 tensors, 1 metadata (alignment)
    data += makeGGUFHeader(tensorCount: 0, metadataCount: 1)

    // Metadata: general.alignment = 32
    data += makeGGUFString("general.alignment")
    data += [0x04, 0x00, 0x00, 0x00]  // uint32
    data += [0x20, 0x00, 0x00, 0x00]  // 32

    // Calculate: 24 (header) + 8 + 17 + 4 + 4 = 57 bytes
    // Padding to 32: 32 - (57 % 32) = 32 - 25 = 7 bytes
    data += [UInt8](repeating: 0x00, count: 7)

    let gguf = try GGUF(parsing: data)

    #expect(gguf.header.version == 3)
    #expect(gguf.header.tensorCount == 0)
    #expect(gguf.header.metadataKeyValueCount == 1)
    #expect(gguf.metadata.count == 1)
    #expect(gguf.metadata[0].key == "general.alignment")
    #expect(gguf.tensorInfos.count == 0)
    #expect(gguf.alignment == 32)
}

@Test func `parsing data with tensors`() throws {
    var data: [UInt8] = []

    // Header: 2 tensors, 1 metadata
    data += makeGGUFHeader(tensorCount: 2, metadataCount: 1)

    // Metadata: general.alignment = 32
    data += makeGGUFString("general.alignment")
    data += [0x04, 0x00, 0x00, 0x00]  // uint32
    data += [0x20, 0x00, 0x00, 0x00]  // 32

    // Tensor 1: "layer1.weight"
    data += makeGGUFString("layer1.weight")
    data += [0x02, 0x00, 0x00, 0x00]  // 2 dimensions
    data += littleEndianBytes(UInt64(512))
    data += littleEndianBytes(UInt64(256))
    data += [0x00, 0x00, 0x00, 0x00]  // F32
    data += [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]  // offset 0

    // Tensor 2: "layer1.bias"
    data += makeGGUFString("layer1.bias")
    data += [0x01, 0x00, 0x00, 0x00]  // 1 dimension
    data += littleEndianBytes(UInt64(512))
    data += [0x00, 0x00, 0x00, 0x00]  // F32
    data += littleEndianBytes(UInt64(512 * 256 * 4))  // offset after layer1.weight

    // Calculate padding
    let headerSize = 24  // Fixed: header is 24 bytes, not 32
    let metadataSize = 8 + 17 + 4 + 4  // length + key + type + value
    let tensor1Size = 8 + 13 + 4 + 16 + 4 + 8  // length + name + dimCount + dims + type + offset
    let tensor2Size = 8 + 11 + 4 + 8 + 4 + 8  // length + name + dimCount + dims + type + offset
    let totalSize = headerSize + metadataSize + tensor1Size + tensor2Size
    let paddingNeeded = (32 - (totalSize % 32)) % 32
    data += [UInt8](repeating: 0x00, count: paddingNeeded)

    let gguf = try GGUF(parsing: data)

    #expect(gguf.header.tensorCount == 2)
    #expect(gguf.tensorInfos.count == 2)
    #expect(gguf.tensorInfos[0].name == "layer1.weight")
    #expect(gguf.tensorInfos[0].dimensions == [512, 256])
    #expect(gguf.tensorInfos[1].name == "layer1.bias")
    #expect(gguf.tensorInfos[1].dimensions == [512])
}

@Test func `parsing metadata`() throws {
    var data: [UInt8] = []

    // Header: 0 tensors, 2 metadata
    data += makeGGUFHeader(tensorCount: 0, metadataCount: 2)

    // Metadata 1: general.name = "test_model"
    data += makeGGUFString("general.name")
    data += [0x08, 0x00, 0x00, 0x00]  // string
    data += makeGGUFString("test_model")

    // Metadata 2: general.alignment = 32
    data += makeGGUFString("general.alignment")
    data += [0x04, 0x00, 0x00, 0x00]  // uint32
    data += [0x20, 0x00, 0x00, 0x00]  // 32

    // Calculate and add padding
    // Header: 24, Metadata1: 8+12+4+8+10=42, Metadata2: 8+17+4+4=33, Total: 99
    let totalSize = 24 + 42 + 33  // = 99
    let paddingNeeded = (32 - (totalSize % 32)) % 32  // = 29
    data += [UInt8](repeating: 0x00, count: paddingNeeded)

    let gguf = try GGUF(parsing: data)

    let nameValue = try #require(gguf.metadataValue(forKey: "general.name"))
    #expect(isClose(nameValue, .string("test_model")))

    let alignmentValue = try #require(gguf.metadataValue(forKey: "general.alignment"))
    #expect(isClose(alignmentValue, .uint32(32)))

    let nonExistent = gguf.metadataValue(forKey: "does.not.exist")
    #expect(nonExistent == nil)
}

@Test func `parsing invalid padding should throw error`() throws {
    var data: [UInt8] = []

    // Header: 0 tensors, 1 metadata
    data += makeGGUFHeader(tensorCount: 0, metadataCount: 1)

    // Metadata: general.alignment = 32
    data += makeGGUFString("general.alignment")
    data += [0x04, 0x00, 0x00, 0x00]  // uint32
    data += [0x20, 0x00, 0x00, 0x00]  // 32

    // Invalid padding (non-zero bytes)
    // Header (24) + metadata (8+17+4+4=33) = 57 bytes, need 7 bytes padding
    data += [UInt8](repeating: 0xFF, count: 7)  // Should be 0x00

    #expect(throws: GGUF.Error.self) {
        _ = try GGUF(parsing: data)
    }
}

@Test func `parsing default alignment`() throws {
    var data: [UInt8] = []

    // Header: 0 tensors, 0 metadata (no alignment specified)
    data += makeGGUFHeader(tensorCount: 0, metadataCount: 0)

    // Header is 24 bytes, need 8 bytes padding to reach 32
    data += [UInt8](repeating: 0x00, count: 8)

    let gguf = try GGUF(parsing: data)

    // Should default to 32
    #expect(gguf.alignment == 32)
}

@Test func `tensor data extraction`() throws {
    var data: [UInt8] = []

    // Header
    data += makeGGUFHeader(tensorCount: 1, metadataCount: 1)

    // Metadata: general.alignment = 32
    data += makeGGUFString("general.alignment")
    data += [0x04, 0x00, 0x00, 0x00]  // Type: uint32
    data += [0x20, 0x00, 0x00, 0x00]  // Value: 32

    // Tensor info: 2x3 F32 tensor (6 elements × 4 bytes = 24 bytes)
    data += makeGGUFString("test.tensor")
    data += [0x02, 0x00, 0x00, 0x00]  // Dimension count: 2
    data += littleEndianBytes(UInt64(2))  // Dimension 0: 2
    data += littleEndianBytes(UInt64(3))  // Dimension 1: 3
    data += [0x00, 0x00, 0x00, 0x00]  // Data type: F32 (0)
    data += [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]  // Offset: 0

    // Calculate alignment padding
    // Header: 24, Metadata: 8+17+4+4=33, Tensor: 8+11+4+8+8+4+8=51 = 108 total
    let currentOffset = data.count
    let alignment = 32
    let paddingNeeded = (alignment - (currentOffset % alignment)) % alignment
    data += Array(repeating: UInt8(0), count: paddingNeeded)

    // Tensor data: 6 float32 values
    let tensorValues: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    for value in tensorValues {
        data += littleEndianBytes(value.bitPattern)
    }

    // Parse and extract
    let fileData = Data(data)
    let gguf = try GGUF(parsing: fileData)

    // Extract tensor data
    let extractedData = gguf.tensorData(at: 0, from: fileData)

    // Verify size
    #expect(extractedData.count == 24)  // 6 floats × 4 bytes

    // Verify values
    let extractedFloats = extractedData.withUnsafeBytes { buffer in
        Array(buffer.bindMemory(to: Float.self))
    }
    #expect(allClose(extractedFloats, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
}

@Test func `tensor header and metadata extraction with real file`() throws {
    let testFileURL = try #require(
        Bundle.module.url(
            forResource: "small",
            withExtension: "gguf",
            subdirectory: "Resources"
        )
    )
    let fileData = try Data(contentsOf: testFileURL)
    let gguf = try GGUF(parsing: fileData)

    #expect(gguf.metadata.count == Int(gguf.header.metadataKeyValueCount))
    #expect(gguf.tensorInfos.count == 10)
    #expect(gguf.tensorInfos.count == Int(gguf.header.tensorCount))
    #expect(!gguf.metadata.isEmpty)
}

@Test func `tensor data extraction with real file`() throws {
    let testFileURL = try #require(
        Bundle.module.url(
            forResource: "small",
            withExtension: "gguf",
            subdirectory: "Resources"
        )
    )
    let fileData = try Data(contentsOf: testFileURL)
    let gguf = try GGUF(parsing: fileData)

    let tensor1 = gguf.tensorInfos[0]
    let tensorData1 = gguf.tensorData(at: 0, from: fileData)
    let floatArray1 = try gguf.tensorFloatArray(at: 0, from: fileData)
    #expect(tensorData1.count == tensor1.sizeInBytes)
    #expect(gguf.tensorData("tensor_0", from: fileData) == tensorData1)
    try #expect(allClose(gguf.tensorFloatArray("tensor_0", from: fileData) ?? [], floatArray1))

    let tensor2 = gguf.tensorInfos[1]
    let tensorData2 = gguf.tensorData(at: 1, from: fileData)
    let floatArray2 = try gguf.tensorFloatArray(at: 1, from: fileData)
    #expect(tensorData2.count == tensor2.sizeInBytes)
    #expect(gguf.tensorData("tensor_1", from: fileData) == tensorData2)
    try #expect(allClose(gguf.tensorFloatArray("tensor_1", from: fileData) ?? [], floatArray2))

    try #expect(gguf.tensorFloatArray("other name", from: fileData) == nil)
    #expect(gguf.tensorData("other name", from: fileData) == nil)
}
