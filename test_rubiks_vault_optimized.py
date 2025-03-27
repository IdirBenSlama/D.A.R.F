#!/usr/bin/env python3
"""
Test module for Rubik's Vault - Optimized Version

This module provides tests for the Rubik's Vault optimized implementation,
focusing specifically on validating fixes for identified issues including:
- Atomic file operations for metadata updates
- Proper error handling
- Memory management and resource cleanup
- Concurrency safety
"""

import os
import sys
import json
import time
import asyncio
import logging
import tempfile
import threading
import unittest
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

# Add parent directory to path to import DARF modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from src.modules.rubiks_vault.rubiks_vault_optimized import (
        ContentStore, SecretManager, StorageTier, AccessLevel, 
        EncryptionMode, CachePolicy, CryptoHelper
    )
except ImportError as e:
    print(f"Error importing Rubik's Vault modules: {e}")
    # Create mock classes for testing without the actual implementation
    class StorageTier:
        HOT = "hot"
        WARM = "warm"
        COLD = "cold"
    
    class AccessLevel:
        PUBLIC = "public"
        RESTRICTED = "restricted"
        PRIVATE = "private"
        
    class EncryptionMode:
        NONE = "none"
        SIMPLE = "simple"
        ADVANCED = "advanced"
        
    class CachePolicy:
        NONE = "none"
        LRU = "lru"
        TTL = "ttl"
        
    class ContentStore:
        def __init__(self, *args, **kwargs):
            pass
            
    class SecretManager:
        def __init__(self, *args, **kwargs):
            pass
            
    class CryptoHelper:
        def __init__(self, *args, **kwargs):
            pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RubiksVaultTest")


class TestDataCorruption(unittest.TestCase):
    """Tests for preventing data corruption during file operations."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.content_store = ContentStore(base_path=self.test_dir)
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_atomic_metadata_update(self):
        """Test that metadata updates are atomic."""
        # Store some initial content
        test_data = b"test content"
        content_hash = self.content_store.store(test_data)
        
        # Get initial metadata
        initial_metadata = self.content_store.get_metadata(content_hash)
        
        # Create a function that simulates a crash during update
        def simulate_crash_during_update():
            # Get the metadata path
            meta_path = self.content_store._get_meta_path(content_hash)
            temp_path = meta_path.with_suffix('.tmp')
            
            # Monitor for temp file creation
            start_time = time.time()
            while not temp_path.exists() and time.time() - start_time < 5:
                time.sleep(0.01)
            
            # If temp file exists, simulate crash by terminating the update thread
            if temp_path.exists():
                # Force Python to "crash" the update thread
                update_thread.terminate()
        
        # Update metadata in a separate thread
        class TerminatableThread(threading.Thread):
            def terminate(self):
                self._terminated = True
                
            def run(self):
                self._terminated = False
                while not self._terminated:
                    try:
                        # Update metadata with new values
                        new_metadata = initial_metadata.copy()
                        new_metadata["test_key"] = "test_value"
                        self.content_store._update_metadata(content_hash, new_metadata)
                        break
                    except:
                        # Thread terminated during update
                        pass
        
        # Create and start the update thread
        update_thread = TerminatableThread(target=lambda: None)
        update_thread.start()
        
        # Simulate crash
        simulate_crash_during_update()
        
        # Wait for the thread to complete
        update_thread.join(timeout=1)
        
        # Verify metadata is either completely updated or not changed at all
        current_metadata = self.content_store.get_metadata(content_hash)
        
        # Should either be the same as initial or completely updated
        self.assertTrue(
            current_metadata == initial_metadata or
            (current_metadata != initial_metadata and 
             "test_key" in current_metadata and 
             current_metadata["test_key"] == "test_value")
        )
        
        # Check that no temp files are left behind
        temp_path = Path(self.content_store._get_meta_path(content_hash)).with_suffix('.tmp')
        self.assertFalse(temp_path.exists())


class TestConcurrencyIssues(unittest.TestCase):
    """Tests for concurrent access to the Rubik's Vault."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.content_store = ContentStore(base_path=self.test_dir)
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_concurrent_store_retrieve(self):
        """Test concurrent store and retrieve operations."""
        # Number of threads to use
        NUM_THREADS = 10
        # Number of operations per thread
        OPS_PER_THREAD = 20
        
        # Content to store
        test_data = [f"test content {i}".encode() for i in range(OPS_PER_THREAD)]
        
        # Track hashes and success counts
        stored_hashes = []
        successful_stores = [0]
        successful_retrieves = [0]
        
        def store_worker():
            """Worker function to store content."""
            for i in range(OPS_PER_THREAD):
                try:
                    hash = self.content_store.store(test_data[i])
                    stored_hashes.append(hash)
                    successful_stores[0] += 1
                except Exception as e:
                    logger.error(f"Error storing content: {e}")
                    
        def retrieve_worker():
            """Worker function to retrieve content."""
            for _ in range(OPS_PER_THREAD):
                if not stored_hashes:
                    time.sleep(0.01)
                    continue
                    
                # Get a random hash from stored_hashes
                try:
                    hash = stored_hashes[hash_index % len(stored_hashes)]
                    content = self.content_store.retrieve(hash)
                    if content is not None:
                        successful_retrieves[0] += 1
                except Exception as e:
                    logger.error(f"Error retrieving content: {e}")
                hash_index += 1
                
        # Create and start store threads
        store_threads = [threading.Thread(target=store_worker) for _ in range(NUM_THREADS)]
        for thread in store_threads:
            thread.start()
            
        # Give store threads a head start
        time.sleep(0.1)
        
        # Create and start retrieve threads
        hash_index = 0
        retrieve_threads = [threading.Thread(target=retrieve_worker) for _ in range(NUM_THREADS)]
        for thread in retrieve_threads:
            thread.start()
            
        # Wait for all threads to complete
        for thread in store_threads + retrieve_threads:
            thread.join()
            
        # Verify results
        self.assertEqual(successful_stores[0], NUM_THREADS * OPS_PER_THREAD, 
                         "Not all store operations succeeded")
        self.assertGreater(successful_retrieves[0], 0, 
                           "No retrieve operations succeeded")


class TestResourceManagement(unittest.TestCase):
    """Tests for proper resource management and cleanup."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_file_handle_cleanup(self):
        """Test that file handles are properly cleaned up."""
        # Create ContentStore
        content_store = ContentStore(base_path=self.test_dir)
        
        # Store some data
        test_data = b"test content for file handle test"
        content_hash = content_store.store(test_data)
        
        # Get the data path
        data_path = content_store._get_data_path(content_hash)
        
        # Check if file is locked or can be opened for writing
        try:
            with open(data_path, 'rb+') as f:
                # Should be able to open file after ContentStore operations are done
                pass
            can_open = True
        except IOError:
            can_open = False
            
        self.assertTrue(can_open, "File handle was not properly closed")


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling in Rubik's Vault."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.content_store = ContentStore(base_path=self.test_dir)
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_invalid_content_hash(self):
        """Test handling of invalid content hash."""
        # Try to retrieve non-existent content
        result = self.content_store.retrieve("nonexistent_hash")
        self.assertIsNone(result, "Should return None for non-existent content")
        
        # Try to get metadata for non-existent content
        result = self.content_store.get_metadata("nonexistent_hash")
        self.assertIsNone(result, "Should return None for non-existent metadata")
        

class TestSecretManager(unittest.TestCase):
    """Tests for SecretManager implementation."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        config_path = os.path.join(self.test_dir, "secrets")
        os.makedirs(config_path, exist_ok=True)
        self.secret_manager = SecretManager(config_path=config_path)
        self.secret_manager.initialize()
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_secret_store_retrieve(self):
        """Test storing and retrieving secrets."""
        # Store a secret
        self.secret_manager.store_secret("test_secret", "test value")
        
        # Retrieve the secret
        value = self.secret_manager.retrieve_secret("test_secret")
        self.assertEqual(value, "test value", "Retrieved secret doesn't match stored value")
        
        # List secrets
        secrets = self.secret_manager.list_secrets()
        self.assertIn("test_secret", secrets, "Stored secret not found in list")
        
        # Delete a secret
        self.secret_manager.delete_secret("test_secret")
        
        # Verify it's gone
        value = self.secret_manager.retrieve_secret("test_secret")
        self.assertIsNone(value, "Secret should be None after deletion")


class TestCryptoImplementation(unittest.TestCase):
    """Tests for CryptoHelper implementation."""
    
    def setUp(self):
        """Set up test environment."""
        self.crypto = CryptoHelper()
        
    def test_key_generation(self):
        """Test key generation."""
        key = self.crypto.generate_key()
        self.assertEqual(len(key), 32, "Generated key should be 32 bytes")
        
    def test_key_splitting_reconstruction(self):
        """Test key splitting and reconstruction."""
        key = self.crypto.generate_key()
        shares = self.crypto.split_key(key)
        self.assertEqual(len(shares), 3, "Should generate 3 shares by default")
        
        # Reconstruct from all shares
        reconstructed = self.crypto.reconstruct_key(shares)
        self.assertEqual(key, reconstructed, "Reconstructed key should match original")
        
        # Reconstruct from minimum shares
        reconstructed = self.crypto.reconstruct_key(shares[:2])
        self.assertEqual(key, reconstructed, "Should reconstruct from minimum shares")
        
    def test_encryption_decryption(self):
        """Test encryption and decryption."""
        key = self.crypto.generate_key()
        data = b"test data for encryption"
        
        # Encrypt data
        encrypted = self.crypto.encrypt(key, data)
        self.assertNotEqual(data, encrypted, "Encrypted data should differ from original")
        
        # Decrypt data
        decrypted = self.crypto.decrypt(key, encrypted)
        self.assertEqual(data, decrypted, "Decrypted data should match original")
    
    def test_key_derivation(self):
        """Test key derivation for different purposes."""
        master_key = self.crypto.generate_key()
        
        # Derive keys for different purposes
        key1 = self.crypto.derive_key(master_key, "purpose1")
        key2 = self.crypto.derive_key(master_key, "purpose2")
        
        # Keys should be different
        self.assertNotEqual(key1, key2, "Derived keys should be different")
        
        # Keys should be deterministic
        key1_again = self.crypto.derive_key(master_key, "purpose1")
        self.assertEqual(key1, key1_again, "Key derivation should be deterministic")


if __name__ == "__main__":
    unittest.main()
