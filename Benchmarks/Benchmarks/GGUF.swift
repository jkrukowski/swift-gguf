import Benchmark
import Foundation
import GGUF

func loadData() -> Data {
    let url = Bundle.module.url(
        forResource: "tinyllama-1.1b-chat-v1.0.Q2_K",
        withExtension: "gguf",
        subdirectory: "Data"
    )!
    return try! Data(contentsOf: url, options: .mappedIfSafe)
}

let benchmarks: @Sendable () -> Void = {
    Benchmark("Parse Q2_K GGUF") { benchmark in
        let data = loadData()
        benchmark.startMeasurement()
        for _ in benchmark.scaledIterations {
            blackHole(try GGUF(parsing: data))
        }
    }

    Benchmark("Load Q2_K float array") { benchmark in
        let data = loadData()
        let gguf = try GGUF(parsing: data)
        benchmark.startMeasurement()
        for _ in benchmark.scaledIterations {
            blackHole(try gguf.tensorFloatArray(at: 0, from: data))
        }
    }
}
