"""
Security utilities for NestAI.
"""

import re
import os
import json
import base64
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set, Pattern
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class PIIDetector:
    """
    Detects and redacts personally identifiable information (PII).
    """
    
    def __init__(self, custom_patterns: Optional[Dict[str, str]] = None):
        """
        Initialize the PII detector.
        
        Args:
            custom_patterns: Custom regex patterns for PII detection
        """
        self.patterns: Dict[str, Pattern] = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b(?:\+\d{1,3}[- ]?)?$$?\d{3}$$?[- ]?\d{3}[- ]?\d{4}\b'),
            "ssn": re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'),
            "credit_card": re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b'),
            "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
            "date_of_birth": re.compile(r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b')
        }
        
        # Add custom patterns
        if custom_patterns:
            for name, pattern in custom_patterns.items():
                self.patterns[name] = re.compile(pattern)
    
    def detect(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII in text.
        
        Args:
            text: The text to check
            
        Returns:
            A list of detected PII items
        """
        results = []
        
        for pii_type, pattern in self.patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                results.append({
                    "type": pii_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Sort by position
        results.sort(key=lambda x: x["start"])
        
        return results
    
    def redact(self, text: str, replacement: str = "[REDACTED]") -> Tuple[str, List[Dict[str, Any]]]:
        """
        Detect and redact PII in text.
        
        Args:
            text: The text to check
            replacement: The replacement text
            
        Returns:
            A tuple of (redacted_text, detected_pii)
        """
        detected = self.detect(text)
        
        # Redact from end to start to avoid position shifts
        redacted_text = text
        for item in reversed(detected):
            redacted_text = redacted_text[:item["start"]] + replacement + redacted_text[item["end"]:]
        
        return redacted_text, detected


class DataEncryptor:
    """
    Encrypts and decrypts sensitive data.
    """
    
    def __init__(self, key_file: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize the data encryptor.
        
        Args:
            key_file: Path to the key file
            password: Password for key derivation
        """
        self.key = self._get_or_create_key(key_file, password)
        self.fernet = Fernet(self.key)
    
    def _get_or_create_key(self, key_file: Optional[str], password: Optional[str]) -> bytes:
        """
        Get or create an encryption key.
        
        Args:
            key_file: Path to the key file
            password: Password for key derivation
            
        Returns:
            The encryption key
        """
        if key_file and os.path.exists(key_file):
            # Load existing key
            with open(key_file, "rb") as f:
                return f.read()
        
        # Generate a new key
        if password:
            # Derive key from password
            salt = b'nestai_salt'  # In a real app, use a random salt and store it
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        else:
            # Generate random key
            key = Fernet.generate_key()
        
        # Save key if key_file is provided
        if key_file:
            os.makedirs(os.path.dirname(key_file), exist_ok=True)
            with open(key_file, "wb") as f:
                f.write(key)
        
        return key
    
    def encrypt(self, data: str) -> str:
        """
        Encrypt data.
        
        Args:
            data: The data to encrypt
            
        Returns:
            The encrypted data
        """
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt data.
        
        Args:
            encrypted_data: The encrypted data
            
        Returns:
            The decrypted data
        """
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_dict(self, data: Dict[str, Any], sensitive_keys: Set[str]) -> Dict[str, Any]:
        """
        Encrypt sensitive fields in a dictionary.
        
        Args:
            data: The dictionary
            sensitive_keys: Set of sensitive keys
            
        Returns:
            The dictionary with encrypted fields
        """
        result = {}
        
        for key, value in data.items():
            if key in sensitive_keys:
                if isinstance(value, str):
                    result[key] = self.encrypt(value)
                elif isinstance(value, dict):
                    result[key] = self.encrypt_dict(value, sensitive_keys)
                else:
                    result[key] = value
            elif isinstance(value, dict):
                result[key] = self.encrypt_dict(value, sensitive_keys)
            else:
                result[key] = value
        
        return result
    
    def decrypt_dict(self, data: Dict[str, Any], sensitive_keys: Set[str]) -> Dict[str, Any]:
        """
        Decrypt sensitive fields in a dictionary.
        
        Args:
            data: The dictionary
            sensitive_keys: Set of sensitive keys
            
        Returns:
            The dictionary with decrypted fields
        """
        result = {}
        
        for key, value in data.items():
            if key in sensitive_keys:
                if isinstance(value, str):
                    try:
                        result[key] = self.decrypt(value)
                    except Exception:
                        # If decryption fails, keep the original value
                        result[key] = value
                elif isinstance(value, dict):
                    result[key] = self.decrypt_dict(value, sensitive_keys)
                else:
                    result[key] = value
            elif isinstance(value, dict):
                result[key] = self.decrypt_dict(value, sensitive_keys)
            else:
                result[key] = value
        
        return result


class SecurityAuditor:
    """
    Audits security events.
    """
    
    def __init__(self, audit_dir: Optional[str] = None):
        """
        Initialize the security auditor.
        
        Args:
            audit_dir: Directory for storing audit logs
        """
        self.audit_dir = audit_dir
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log a security event.
        
        Args:
            event_type: The type of event
            data: Event data
        """
        if not self.audit_dir:
            return
        
        os.makedirs(self.audit_dir, exist_ok=True)
        
        # Create event record
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        
        # Generate filename based on date
        date_str = datetime.now().strftime("%Y%m%d")
        filename = os.path.join(self.audit_dir, f"audit_{date_str}.jsonl")
        
        # Append event to file
        with open(filename, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get security events.
        
        Args:
            start_date: Start date
            end_date: End date
            event_types: Event types to include
            
        Returns:
            A list of events
        """
        if not self.audit_dir or not os.path.exists(self.audit_dir):
            return []
        
        events = []
        
        # Get list of audit files
        audit_files = [f for f in os.listdir(self.audit_dir) if f.startswith("audit_") and f.endswith(".jsonl")]
        
        for filename in audit_files:
            file_path = os.path.join(self.audit_dir, filename)
            
            with open(file_path, "r") as f:
                for line in f:
                    event = json.loads(line)
                    
                    # Apply filters
                    event_timestamp = datetime.fromisoformat(event["timestamp"])
                    
                    if start_date and event_timestamp < start_date:
                        continue
                    
                    if end_date and event_timestamp > end_date:
                        continue
                    
                    if event_types and event["event_type"] not in event_types:
                        continue
                    
                    events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda x: x["timestamp"])
        
        return events

