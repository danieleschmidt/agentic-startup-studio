import pytest

from tests.framework.data_manager import (
    DataManager,
    create_edge_case_test_data,
    create_malicious_payload_data,
)


class TestSyntheticDataGeneration:
    def test_should_generate_edge_case_data(self):
        manager = DataManager()
        dataset = manager.generate_synthetic_data('edge_cases', 3)
        assert dataset.schema['type'] == 'edge_cases'
        assert dataset.data['count'] == 3
        for record in dataset.data['records']:
            assert record['market_size'] == 0
            assert record['implementation_complexity'] == 0

    def test_should_generate_malicious_payloads(self):
        manager = DataManager()
        dataset = manager.generate_synthetic_data('malicious_payloads', 2)
        assert dataset.schema['type'] == 'malicious_payloads'
        assert dataset.data['count'] == 2
        for record in dataset.data['records']:
            assert 'payload' in record

    def test_factory_functions(self):
        edge_dataset = create_edge_case_test_data(1)
        malicious_dataset = create_malicious_payload_data(1)
        assert edge_dataset.schema['type'] == 'edge_cases'
        assert malicious_dataset.schema['type'] == 'malicious_payloads'
