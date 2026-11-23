import SwiftUI
import UIKit

@main
struct SmartScoutProApp: App {
    init() {
        // Keep iPhone screen awake during continuous tripod radar-gun operation
        UIApplication.shared.isIdleTimerDisabled = true
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
        }
    }
}
