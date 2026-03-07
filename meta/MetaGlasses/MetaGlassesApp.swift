import SwiftUI
import MWDATCore

@main
struct MetaGlassesApp: App {
    @StateObject private var glassesManager = GlassesManager()

    init() {
        // Must be called once at launch before any other SDK usage
        do {
            try Wearables.configure()
        } catch {
            print("Wearables SDK configure failed: \(error)")
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(glassesManager)
                // The Meta AI app deep-links back here after registration/permissions
                .onOpenURL { url in
                    Task {
                        try? await glassesManager.handleCallback(url: url)
                    }
                }
        }
    }
}
