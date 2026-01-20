"""
Project Analyzer for RLM Processing

Extracts project structure and key files from metadata files like:
- CLAUDE.md, README.md (documentation)
- Package.swift, *.xcodeproj (iOS)
- package.json, tsconfig.json (JS/TS)
- Cargo.toml, pyproject.toml (Rust/Python)

This helps RLM understand project context and find important files.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from .file_collector import CollectionResult


@dataclass
class ProjectInfo:
    """Extracted project information."""
    project_type: str = "unknown"  # ios, web, python, rust, etc.
    key_files: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    important_directories: list[str] = field(default_factory=list)
    mentioned_features: list[str] = field(default_factory=list)
    tech_stack: list[str] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Convert to context string for LLM."""
        parts = [f"## Project Context ({self.project_type})"]

        if self.key_files:
            parts.append(f"\n### Key Files (from documentation)")
            for f in self.key_files[:20]:
                parts.append(f"  - {f}")

        if self.entry_points:
            parts.append(f"\n### Entry Points")
            for e in self.entry_points[:10]:
                parts.append(f"  - {e}")

        if self.tech_stack:
            parts.append(f"\n### Tech Stack")
            parts.append(f"  {', '.join(self.tech_stack[:15])}")

        if self.mentioned_features:
            parts.append(f"\n### Mentioned Features")
            for f in self.mentioned_features[:15]:
                parts.append(f"  - {f}")

        return "\n".join(parts)


class ProjectAnalyzer:
    """
    Analyzes project metadata to extract structure and key files.

    Reads documentation and config files to understand:
    - Project type (iOS, web, Python, etc.)
    - Key files and directories
    - Tech stack
    - Important features mentioned
    """

    # Metadata files to look for (priority order)
    METADATA_FILES = [
        "CLAUDE.md",
        "README.md",
        "Package.swift",
        "project.pbxproj",
        "package.json",
        "tsconfig.json",
        "Cargo.toml",
        "pyproject.toml",
        "setup.py",
        "go.mod",
        "build.gradle",
        "pom.xml",
        # iOS-specific config files
        "Info.plist",
        "Podfile",
        "Cartfile",
        ".xcconfig",
    ]

    # iOS-specific indicator files and directories
    IOS_INDICATORS = [
        "Package.swift",
        "project.pbxproj",
        "Info.plist",
        "Podfile",
        "Cartfile",
        "AppDelegate.swift",
        "SceneDelegate.swift",
        "ContentView.swift",
        ".entitlements",
        ".xcconfig",
        ".xcassets",
        "LaunchScreen.storyboard",
    ]

    # Patterns to extract file references from documentation
    FILE_PATTERNS = [
        r'`([a-zA-Z0-9_/\-\.]+\.[a-zA-Z]{1,10})`',  # `path/to/file.ext`
        r'\*\*([a-zA-Z0-9_/\-\.]+\.[a-zA-Z]{1,10})\*\*',  # **path/to/file.ext**
        r'- ([a-zA-Z0-9_/\-\.]+\.[a-zA-Z]{1,10})',  # - path/to/file.ext
        r'([A-Z][a-zA-Z]+(?:View|ViewController|Controller|Manager|Service|Model|Handler|Coordinator|Router|Repository|UseCase|Interactor)\.swift)',  # SwiftUIView.swift
        r'([a-z_]+\.py)',  # python_file.py
        r'([A-Z][a-zA-Z]+(?:App|Scene)\.swift)',  # App entry points
    ]

    # Directory patterns
    DIRECTORY_PATTERNS = [
        r'`(src/[a-zA-Z0-9_/\-]+)`',
        r'`(lib/[a-zA-Z0-9_/\-]+)`',
        r'`(\./[a-zA-Z0-9_/\-]+)`',
    ]

    # Feature/capability patterns
    FEATURE_PATTERNS = [
        r'(?:implements?|features?|supports?|includes?)\s+([A-Za-z][A-Za-z0-9\s]{2,30})',
        r'(?:StoreKit|CloudKit|HealthKit|CoreData|SwiftUI|UIKit)',
        r'(?:authentication|authorization|payment|subscription|sync)',
    ]

    def analyze(self, collection: CollectionResult) -> ProjectInfo:
        """
        Analyze project metadata from collected files.

        Args:
            collection: The file collection to analyze

        Returns:
            ProjectInfo with extracted information
        """
        info = ProjectInfo()

        # Find and analyze metadata files
        metadata_content = {}
        for f in collection.files:
            filename = f.relative_path.split('/')[-1]
            if filename in self.METADATA_FILES or any(f.relative_path.endswith(m) for m in self.METADATA_FILES):
                metadata_content[filename] = f.content

        # Detect project type
        info.project_type = self._detect_project_type(metadata_content, collection)

        # Extract key files from documentation
        if "CLAUDE.md" in metadata_content:
            info.key_files.extend(self._extract_file_refs(metadata_content["CLAUDE.md"]))
            info.mentioned_features.extend(self._extract_features(metadata_content["CLAUDE.md"]))

        if "README.md" in metadata_content:
            info.key_files.extend(self._extract_file_refs(metadata_content["README.md"]))
            info.mentioned_features.extend(self._extract_features(metadata_content["README.md"]))

        # Extract from package managers
        if "Package.swift" in metadata_content:
            info.tech_stack.append("Swift Package Manager")
            info.entry_points.extend(self._extract_swift_targets(metadata_content["Package.swift"]))

        if "package.json" in metadata_content:
            pkg_info = self._parse_package_json(metadata_content["package.json"])
            info.tech_stack.extend(pkg_info.get("dependencies", [])[:10])
            info.entry_points.append(pkg_info.get("main", "index.js"))

        if "pyproject.toml" in metadata_content or "setup.py" in metadata_content:
            info.tech_stack.append("Python")

        if "Cargo.toml" in metadata_content:
            info.tech_stack.append("Rust")

        # Deduplicate
        info.key_files = list(dict.fromkeys(info.key_files))
        info.mentioned_features = list(dict.fromkeys(info.mentioned_features))
        info.tech_stack = list(dict.fromkeys(info.tech_stack))

        return info

    def _detect_project_type(
        self,
        metadata: dict[str, str],
        collection: CollectionResult
    ) -> str:
        """Detect project type from metadata and file extensions."""
        # Check metadata files first
        if "Package.swift" in metadata or "project.pbxproj" in metadata:
            return "ios"
        if "Info.plist" in metadata or "Podfile" in metadata or "Cartfile" in metadata:
            return "ios"
        if "Cargo.toml" in metadata:
            return "rust"
        if "go.mod" in metadata:
            return "go"
        if "pyproject.toml" in metadata or "setup.py" in metadata:
            return "python"

        # Check file names for iOS indicators
        file_names = [f.relative_path.split('/')[-1] for f in collection.files]
        ios_indicator_count = sum(
            1 for indicator in self.IOS_INDICATORS
            if any(indicator in name or name.endswith(indicator) for name in file_names)
        )
        if ios_indicator_count >= 2:
            return "ios"

        # Check file extensions
        extensions = {}
        for f in collection.files:
            ext = f.extension.lower()
            extensions[ext] = extensions.get(ext, 0) + 1

        # Check for iOS-specific extensions
        ios_extensions = extensions.get(".swift", 0) + extensions.get(".m", 0) + extensions.get(".mm", 0)
        ios_config_extensions = (
            extensions.get(".storyboard", 0) +
            extensions.get(".xib", 0) +
            extensions.get(".entitlements", 0) +
            extensions.get(".xcconfig", 0) +
            extensions.get(".plist", 0)
        )

        # Strong iOS signal: Swift files + iOS config files
        if extensions.get(".swift", 0) >= 1 and ios_config_extensions >= 1:
            return "ios"

        # Determine by most common extension
        if ios_extensions > 5:
            return "ios"
        if extensions.get(".ts", 0) + extensions.get(".tsx", 0) > 5:
            return "typescript"
        if extensions.get(".js", 0) + extensions.get(".jsx", 0) > 5:
            return "javascript"
        if extensions.get(".py", 0) > 5:
            return "python"
        if extensions.get(".rs", 0) > 5:
            return "rust"
        if extensions.get(".go", 0) > 5:
            return "go"

        # Even a single Swift file suggests iOS (more aggressive detection)
        if extensions.get(".swift", 0) >= 1:
            return "ios"

        return "unknown"

    def _extract_file_refs(self, content: str) -> list[str]:
        """Extract file references from markdown content."""
        files = []
        for pattern in self.FILE_PATTERNS:
            matches = re.findall(pattern, content)
            files.extend(matches)

        # Filter out likely false positives
        valid_files = []
        for f in files:
            # Must have a reasonable extension
            if '.' in f and len(f) > 3:
                ext = f.split('.')[-1].lower()
                if ext in ['swift', 'py', 'js', 'ts', 'tsx', 'jsx', 'go', 'rs', 'java', 'kt', 'md', 'json', 'yaml', 'toml']:
                    valid_files.append(f)

        return valid_files

    def _extract_features(self, content: str) -> list[str]:
        """Extract mentioned features from documentation."""
        features = []

        # Look for feature keywords - iOS frameworks and common patterns
        feature_keywords = [
            # Apple Frameworks
            'StoreKit', 'CloudKit', 'HealthKit', 'CoreData', 'SwiftUI', 'UIKit',
            'SwiftData', 'Combine', 'CoreML', 'ARKit', 'RealityKit', 'MapKit',
            'CoreLocation', 'CoreMotion', 'CoreBluetooth', 'AVFoundation',
            'PhotosUI', 'Vision', 'NaturalLanguage', 'SpriteKit', 'SceneKit',
            'GameKit', 'PassKit', 'WatchKit', 'WidgetKit', 'AppIntents',
            'ActivityKit', 'WeatherKit', 'MusicKit', 'ShazamKit', 'CallKit',
            'PushKit', 'UserNotifications', 'BackgroundTasks', 'CoreSpotlight',
            # Architecture patterns
            'MVVM', 'MVC', 'VIPER', 'Clean Architecture', 'TCA', 'Coordinator',
            # Common features
            'authentication', 'authorization', 'payment', 'subscription', 'sync',
            'widget', 'extension', 'notification', 'background', 'offline',
            'API', 'REST', 'GraphQL', 'WebSocket', 'database', 'cache',
            'push notification', 'deep link', 'universal link', 'App Clip',
            'iCloud', 'Keychain', 'biometric', 'Face ID', 'Touch ID',
        ]

        content_lower = content.lower()
        for keyword in feature_keywords:
            if keyword.lower() in content_lower:
                features.append(keyword)

        return features

    def _extract_swift_targets(self, content: str) -> list[str]:
        """Extract target names from Package.swift."""
        targets = []
        # Look for .target(name: "X" or .executableTarget(name: "X"
        pattern = r'\.(?:executable)?[Tt]arget\s*\(\s*name:\s*"([^"]+)"'
        matches = re.findall(pattern, content)
        targets.extend(matches)
        return targets

    def _parse_package_json(self, content: str) -> dict[str, Any]:
        """Parse package.json for project info."""
        import json
        try:
            data = json.loads(content)
            result = {
                "main": data.get("main", "index.js"),
                "dependencies": list(data.get("dependencies", {}).keys()),
            }
            return result
        except json.JSONDecodeError:
            return {}

    def get_priority_files(
        self,
        collection: CollectionResult,
        query: str
    ) -> list[str]:
        """
        Get prioritized list of files based on query and project analysis.

        Args:
            collection: File collection
            query: The user's query

        Returns:
            List of file paths, ordered by relevance
        """
        info = self.analyze(collection)
        query_lower = query.lower()

        priority_files = []
        all_files = [f.relative_path for f in collection.files]

        # First, add key files from documentation
        for key_file in info.key_files:
            matches = [f for f in all_files if key_file in f or f.endswith(key_file)]
            priority_files.extend(matches)

        # Then, add files matching query keywords
        query_words = set(query_lower.split())
        for f in all_files:
            f_lower = f.lower()
            # Check if any query word appears in filename
            if any(word in f_lower for word in query_words if len(word) > 3):
                if f not in priority_files:
                    priority_files.append(f)

        # Add based on common patterns
        patterns = {
            "subscription": ["subscription", "payment", "storekit", "iap", "purchase"],
            "widget": ["widget", "extension"],
            "auth": ["auth", "login", "session", "credential"],
            "ui": ["view", "controller", "screen", "component"],
            "data": ["model", "entity", "schema", "database"],
            "api": ["api", "service", "network", "client"],
        }

        for category, keywords in patterns.items():
            if any(kw in query_lower for kw in keywords):
                for f in all_files:
                    f_lower = f.lower()
                    if any(kw in f_lower for kw in keywords):
                        if f not in priority_files:
                            priority_files.append(f)

        # Fill remaining with other files
        for f in all_files:
            if f not in priority_files:
                priority_files.append(f)

        return priority_files
