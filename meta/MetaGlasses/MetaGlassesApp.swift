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
                    print("🔗 Received URL callback: \(url)")
                    guard
                        let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
                        components.queryItems?.contains(where: { $0.name == "metaWearablesAction" }) == true
                    else {
                        print("⚠️ URL not related to DAT SDK, ignoring")
                        return
                    }
                    Task {
                        do {
                            let result = try await Wearables.shared.handleUrl(url)
                            print("✅ URL callback handled: \(result)")
                        } catch {
                            print("❌ URL callback error: \(error)")
                        }
                    }
                }
        }
    }
}
