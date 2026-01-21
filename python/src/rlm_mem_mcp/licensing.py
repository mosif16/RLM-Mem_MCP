#!/usr/bin/env python3
"""
RLM-Mem MCP Commercial Licensing and Usage Tracking
"""

import os
import uuid
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
import httpx


@dataclass
class LicenseInfo:
    """Commercial license information"""
    license_key: str
    organization_name: str
    expiration_date: datetime
    revenue_share_rate: float  # e.g., 0.10 for 10%
    is_valid: bool
    registered_date: datetime
    last_checked: datetime


class LicensingManager:
    """Manages commercial license validation and usage tracking"""

    def __init__(self):
        self.license_server_url = os.getenv("RLM_LICENSE_SERVER_URL", "https://recordandlearn.info/license")
        self.telemetry_enabled = os.getenv("RLM_ENABLE_TELEMETRY", "false").lower() == "true"

    def generate_trial_license(self, organization_name: str) -> str:
        """
        Generate a trial license key for evaluation
        Trial licenses are valid for 30 days with 0% revenue share
        """
        # Create a deterministic key based on organization + timestamp
        seed = f"{organization_name}:{datetime.now().date()}"
        license_hash = hashlib.sha256(seed.encode()).hexdigest()[:24]

        # Format: TRIAL-{org_hash}-{yyyy}-{mm}-{dd}
        trial_key = f"TRIAL-{license_hash}-{datetime.now().strftime('%Y-%m-%d')}"
        return trial_key

    def validate_license(self, license_key: str, organization_name: str) -> LicenseInfo:
        """
        Validate commercial license against remote server
        Returns license information including revenue share rate
        """

        if not license_key:
            # Free use - return dummy license
            return LicenseInfo(
                license_key="",
                organization_name="",
                expiration_date=datetime.max,
                revenue_share_rate=0.0,
                is_valid=False,  # Not a commercial license
                registered_date=datetime.now(),
                last_checked=datetime.now()
            )

        if license_key.startswith("TRIAL-"):
            # Validate trial license
            return self._validate_trial_license(license_key, organization_name)

        # Commercial license validation
        return self._validate_commercial_license(license_key, organization_name)

    def _validate_trial_license(self, license_key: str, organization_name: str) -> LicenseInfo:
        """Validate trial license format and expiration"""
        try:
            parts = license_key.split('-')
            if len(parts) != 5 or parts[0] != 'TRIAL':
                raise ValueError("Invalid trial license format")

            # Extract date from key
            license_date = datetime.strptime(f"{parts[3]}-{parts[4]}", "%Y-%m-%d").date()
            current_date = datetime.now().date()

            # Trial is valid for 30 days from issuance
            is_valid = (current_date - license_date).days <= 30
            expiration_date = datetime.combine(license_date + timedelta(days=30), datetime.min.time())

            return LicenseInfo(
                license_key=license_key,
                organization_name=organization_name,
                expiration_date=expiration_date,
                revenue_share_rate=0.0,  # No revenue share for trial
                is_valid=is_valid,
                registered_date=datetime.combine(license_date, datetime.min.time()),
                last_checked=datetime.now()
            )

        except (ValueError, IndexError):
            return LicenseInfo(
                license_key=license_key,
                organization_name=organization_name,
                expiration_date=datetime.now(),
                revenue_share_rate=0.0,
                is_valid=False,
                registered_date=datetime.now(),
                last_checked=datetime.now()
            )

    def _validate_commercial_license(self, license_key: str, organization_name: str) -> LicenseInfo:
        """Validate commercial license against license server"""
        try:
            # In production, this would make an actual API call
            # For now, return mock validated commercial license
            return LicenseInfo(
                license_key=license_key,
                organization_name=organization_name,
                expiration_date=datetime.now() + timedelta(days=365),  # 1 year validity
                revenue_share_rate=0.10,  # 10% revenue share
                is_valid=True,
                registered_date=datetime.now(),
                last_checked=datetime.now()
            )
        except Exception:
            # If server is unreachable, assume invalid
            return LicenseInfo(
                license_key=license_key,
                organization_name=organization_name,
                expiration_date=datetime.now(),
                revenue_share_rate=0.0,
                is_valid=False,
                registered_date=datetime.now(),
                last_checked=datetime.now()
            )

    def report_usage(self, license_info: LicenseInfo, usage_data: Dict[str, Any]) -> bool:
        """
        Report usage telemetry for compliant license management
        Only sends data if telemetry is enabled
        """
        if not self.telemetry_enabled or not license_info.is_valid:
            return True  # No telemetry = assumed success

        try:
            payload = {
                "license_key": license_info.license_key,
                "organization": license_info.organization_name,
                "timestamp": datetime.now().isoformat(),
                "usage": usage_data
            }

            # In production, send to license server
            # response = httpx.post(f"{self.license_server_url}/api/telemetry", json=payload)
            # return response.status_code == 200

            # For now, just return success
            return True

        except Exception:
            # Fail silently - don't break functionality due to telemetry issues
            return False

    def check_commercial_use_requirement(self, config) -> Optional[str]:
        """
        Check if current configuration requires commercial licensing
        Returns None if ok, or error message if commercial license needed
        """

        # Indicators of commercial use
        indicators = []

        # 1. explicit commercial license key present = commercial use
        if config.commercial_license_key:
            return None  # Has license, ok

        # 2. Organization name suggests commercial
        if config.organization_name and any(keyword in config.organization_name.lower() for keyword in
                                             ['inc', 'llc', 'corp', 'ltd', 'gmbh', 'plc', 'co']):
            indicators.append("organization appears commercial")

        # 3. Environment suggests production/commercial
        if os.getenv("ENVIRONMENT", "").lower() in ['prod', 'production', 'staging']:
            indicators.append("production environment")

        # If commercial indicators present but no license, require commercial license
        if indicators:
            return f"Commercial use detected ({', '.join(indicators)}), commercial license required. Contact licensing@rlm-mem.dev"

        return None  # Free use, no issues


# Global instance for easy access
licensing_manager = LicensingManager()


def validate_commercial_license(config) -> tuple[bool, str, Optional[LicenseInfo]]:
    """
    Validate licensing configuration
    Returns: (is_valid, message, license_info)
    """

    # Check if commercial use is required
    commercial_check = licensing_manager.check_commercial_use_requirement(config)
    if commercial_check:
        return False, commercial_check, None

    # Validate the license
    license_info = licensing_manager.validate_license(
        config.commercial_license_key,
        config.organization_name
    )

    if license_info.is_valid:
        return True, "Commercial license valid", license_info
    elif config.commercial_license_key:
        return False, f"Invalid commercial license: {config.commercial_license_key}", license_info
    else:
        return True, "Free license - ensure non-commercial use", license_info