import Foundation
import Quants
import Testing

@Suite
struct DequantizeEdgeCaseTests {
    @Test
    func `Q4_0 requires correct block size`() {
        // Q4_0 has 32 elements per block
        // Block size: 2 bytes (d) + 16 bytes (qs) = 18 bytes
        let blockSize = 18
        let numBlocks = 10
        let data = Data(count: blockSize * numBlocks)
        let elementCount = 32 * numBlocks

        let result = Dequantize.Q4_0(data, elementCount: elementCount)
        #expect(result.count == elementCount)
    }

    @Test
    func `Q4_1 requires correct block size`() {
        // Q4_1 has 32 elements per block
        // Block size: 2 bytes (d) + 2 bytes (m) + 16 bytes (qs) = 20 bytes
        let blockSize = 20
        let numBlocks = 10
        let data = Data(count: blockSize * numBlocks)
        let elementCount = 32 * numBlocks

        let result = Dequantize.Q4_1(data, elementCount: elementCount)
        #expect(result.count == elementCount)
    }

    @Test
    func `Q5_0 requires correct block size`() {
        // Q5_0 has 32 elements per block
        // Block size: 2 bytes (d) + 4 bytes (qh) + 16 bytes (qs) = 22 bytes
        let blockSize = 22
        let numBlocks = 10
        let data = Data(count: blockSize * numBlocks)
        let elementCount = 32 * numBlocks

        let result = Dequantize.Q5_0(data, elementCount: elementCount)
        #expect(result.count == elementCount)
    }

    @Test
    func `Q5_1 requires correct block size`() {
        // Q5_1 has 32 elements per block
        // Block size: 2 bytes (d) + 2 bytes (m) + 4 bytes (qh) + 16 bytes (qs) = 24 bytes
        let blockSize = 24
        let numBlocks = 10
        let data = Data(count: blockSize * numBlocks)
        let elementCount = 32 * numBlocks

        let result = Dequantize.Q5_1(data, elementCount: elementCount)
        #expect(result.count == elementCount)
    }

    @Test
    func `Q8_0 requires correct block size`() {
        // Q8_0 has 32 elements per block
        // Block size: 2 bytes (d) + 32 bytes (qs) = 34 bytes
        let blockSize = 34
        let numBlocks = 10
        let data = Data(count: blockSize * numBlocks)
        let elementCount = 32 * numBlocks

        let result = Dequantize.Q8_0(data, elementCount: elementCount)
        #expect(result.count == elementCount)
    }

    @Test
    func `IQ4_NL requires correct block size`() {
        // IQ4_NL has 32 elements per block
        // Block size: 2 bytes (d) + 16 bytes (qs) = 18 bytes
        let blockSize = 18
        let numBlocks = 10
        let data = Data(count: blockSize * numBlocks)
        let elementCount = 32 * numBlocks

        let result = Dequantize.IQ4_NL(data, elementCount: elementCount)
        #expect(result.count == elementCount)
    }

    @Test
    func `Q2_K requires correct block size`() {
        // Q2_K has 256 elements per block (QK_K)
        // Block size: 2 bytes (d) + 2 bytes (dmin) + 16 bytes (scales) + 64 bytes (qs) = 84 bytes
        let blockSize = 84
        let numBlocks = 5
        let data = Data(count: blockSize * numBlocks)
        let elementCount = 256 * numBlocks

        let result = Dequantize.Q2_K(data, elementCount: elementCount)
        #expect(result.count == elementCount)
    }

    @Test
    func `Q4_K requires correct block size`() {
        // Q4_K has 256 elements per block (QK_K)
        // Block size: 2 bytes (d) + 2 bytes (dmin) + 12 bytes (scales) + 128 bytes (qs) = 144 bytes
        let blockSize = 144
        let numBlocks = 5
        let data = Data(count: blockSize * numBlocks)
        let elementCount = 256 * numBlocks

        let result = Dequantize.Q4_K(data, elementCount: elementCount)
        #expect(result.count == elementCount)
    }

    @Test
    func `Q6_K requires correct block size`() {
        // Q6_K has 256 elements per block (QK_K)
        // Block size: 2 bytes (d) + 128 bytes (ql) + 64 bytes (qh) + 16 bytes (scales) = 210 bytes
        let blockSize = 210
        let numBlocks = 5
        let data = Data(count: blockSize * numBlocks)
        let elementCount = 256 * numBlocks

        let result = Dequantize.Q6_K(data, elementCount: elementCount)
        #expect(result.count == elementCount)
    }

    @Test
    func `Q4_0 with zero data produces zeros`() {
        let blockSize = 18
        let numBlocks = 2
        let data = Data(count: blockSize * numBlocks)
        let result = Dequantize.Q4_0(data, elementCount: 64)

        // All values should be zero or very close to zero
        for value in result {
            #expect(abs(value) < 0.0001)
        }
    }

    @Test
    func `IQ4_NL with zero data produces zeros`() {
        let blockSize = 18
        let numBlocks = 2
        let data = Data(count: blockSize * numBlocks)
        let result = Dequantize.IQ4_NL(data, elementCount: 64)

        // All values should be zero or very close to zero
        for value in result {
            #expect(abs(value) < 0.0001)
        }
    }

    @Test
    func `Q4_1 single block dequantization`() {
        // Create a single block with known values
        var data = Data()
        // d = 1.0 (fp16)
        data.append(contentsOf: [0x00, 0x3C])  // fp16 for 1.0
        // m = 0.0 (fp16)
        data.append(contentsOf: [0x00, 0x00])
        // qs = 16 bytes of 0x00 (all zeros in 4-bit)
        data.append(Data(count: 16))

        let result = Dequantize.Q4_1(data, elementCount: 32)
        #expect(result.count == 32)

        // With d=1.0, m=0.0, and all quantized values = 0,
        // result should be all zeros
        for value in result {
            #expect(abs(value) < 0.01)
        }
    }

    @Test
    func `IQ4_NL single block with non-linear values`() {
        // Create a single block
        var data = Data()
        // d = 1.0 (fp16)
        data.append(contentsOf: [0x00, 0x3C])  // fp16 for 1.0
        // qs = 16 bytes with specific indices
        // Index 0 maps to -127, Index 15 maps to 113
        data.append(contentsOf: [0xF0])  // indices: 0 (low nibble), 15 (high nibble)
        data.append(Data(count: 15))

        let result = Dequantize.IQ4_NL(data, elementCount: 32)
        #expect(result.count == 32)

        // First value should be close to -127 (index 0)
        #expect(abs(result[0] - (-127.0)) < 1.0)
        // 17th value should be close to 113 (index 15)
        #expect(abs(result[16] - 113.0) < 1.0)
    }

    @Test
    func `Q4_0 multiple blocks consistency`() {
        let blockSize = 18
        let numBlocks = 100
        let data = Data(count: blockSize * numBlocks)
        let result = Dequantize.Q4_0(data, elementCount: 32 * numBlocks)

        #expect(result.count == 32 * numBlocks)
    }

    @Test
    func `IQ4_NL multiple blocks consistency`() {
        let blockSize = 18
        let numBlocks = 100
        let data = Data(count: blockSize * numBlocks)
        let result = Dequantize.IQ4_NL(data, elementCount: 32 * numBlocks)

        #expect(result.count == 32 * numBlocks)
    }

    @Test
    func `Q2_K single block`() {
        let blockSize = 84
        let data = Data(count: blockSize)
        let result = Dequantize.Q2_K(data, elementCount: 256)

        #expect(result.count == 256)
        for value in result {
            #expect(abs(value) < 0.01)
        }
    }

    @Test
    func `Q3_K single block`() {
        let blockSize = 148
        let data = Data(count: blockSize)
        let result = Dequantize.Q3_K(data, elementCount: 256)

        #expect(result.count == 256)
        for value in result {
            #expect(abs(value) < 0.01)
        }
    }

    @Test
    func `Q8_K single block`() {
        let blockSize = 292
        let data = Data(count: blockSize)
        let result = Dequantize.Q8_K(data, elementCount: 256)

        #expect(result.count == 256)
    }
}
