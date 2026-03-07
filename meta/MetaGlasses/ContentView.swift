import SwiftUI

struct ContentView: View {
    @EnvironmentObject var glassesManager: GlassesManager
    @State private var selectedTab: Tab = .identify

    enum Tab {
        case identify, enroll
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 0) {
                // Header
                headerView

                // Live camera preview (always-on during a stream)
                cameraPreview

                // Tab picker
                tabPicker

                // Active tab content
                Group {
                    if selectedTab == .identify {
                        IdentifyView()
                    } else {
                        EnrollView()
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(.horizontal)
                .padding(.bottom, 32)
            }
        }
        .preferredColorScheme(.dark)
        .onAppear {
            // Start streaming as soon as the view appears (requires prior registration)
            glassesManager.startStreamIfReady()
        }
    }

    private var headerView: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("MetaGlasses")
                    .font(.title2.bold())
                    .foregroundColor(.white)
                Text(glassesManager.statusMessage)
                    .font(.caption)
                    .foregroundColor(glassesManager.isStreaming ? .green : .orange)
            }
            Spacer()
            // Register / connect button
            HStack(spacing: 8) {
                if !glassesManager.isRegistered {
                    Button("Connect Glasses") {
                        Task {
                            await glassesManager.register()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.blue)
                    .controlSize(.small)
                    
                    Button("Reset") {
                        Task {
                            await glassesManager.unregister()
                        }
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                    .controlSize(.small)
                } else if glassesManager.isRegistered && !glassesManager.isStreaming {
                    Button("Request Camera") {
                        Task {
                            await glassesManager.requestCameraPermission()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                    .controlSize(.small)
                } else {
                    Image(systemName: "glasses")
                        .foregroundColor(.green)
                        .font(.title2)
                }
            }
        }
        .padding()
    }

    private var cameraPreview: some View {
        ZStack {
            Rectangle()
                .fill(Color.gray.opacity(0.15))

            if let frame = glassesManager.latestFrame {
                Image(uiImage: frame)
                    .resizable()
                    .scaledToFill()
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "camera.fill")
                        .font(.system(size: 36))
                        .foregroundColor(.gray)
                    Text(glassesManager.isRegistered ? "Waiting for stream…" : "Connect glasses first")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
            }
        }
        .frame(height: 240)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .padding(.horizontal)
    }

    private var tabPicker: some View {
        HStack(spacing: 0) {
            tabButton(title: "Identify", icon: "person.fill.questionmark", tab: .identify)
            tabButton(title: "Enroll", icon: "person.badge.plus", tab: .enroll)
        }
        .padding(.horizontal)
        .padding(.vertical, 12)
    }

    private func tabButton(title: String, icon: String, tab: Tab) -> some View {
        Button {
            withAnimation(.easeInOut(duration: 0.2)) { selectedTab = tab }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: icon)
                Text(title)
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
            .background(selectedTab == tab ? Color.blue : Color.clear)
            .foregroundColor(selectedTab == tab ? .white : .gray)
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }
}
