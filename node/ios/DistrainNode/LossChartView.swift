import SwiftUI

/// Simple bar chart showing loss over training steps.
struct LossChartView: View {
    let data: [Double]
    let accentColor: Color

    var body: some View {
        GeometryReader { geo in
            let maxVal = data.max() ?? 1.0
            let minVal = data.min() ?? 0.0
            let range = max(maxVal - minVal, 1e-6)
            let barWidth = max(geo.size.width / CGFloat(data.count), 1)

            ZStack(alignment: .bottomLeading) {
                // Bars
                HStack(alignment: .bottom, spacing: 0) {
                    ForEach(Array(data.enumerated()), id: \.offset) { _, value in
                        let normalized = CGFloat((value - minVal) / range)
                        let height = max(normalized * geo.size.height * 0.9, 1)
                        Rectangle()
                            .fill(accentColor.opacity(0.8))
                            .frame(width: barWidth, height: height)
                    }
                }

                // Y-axis labels
                VStack {
                    Text(String(format: "%.2f", maxVal))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(String(format: "%.2f", minVal))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
        }
    }
}
