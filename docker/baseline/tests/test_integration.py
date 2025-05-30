import pytest
import numpy as np
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from parser import Parser
from detector import Detector, DetectionResult
from integration import Pipeline, IntegrationError


class TestIntegration:
    """Integration test suite for Parser and Detector components"""
    
    @pytest.fixture
    def parser(self):
        """Create parser instance"""
        return Parser()
    
    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        return Detector(threshold=0.5)
    
    @pytest.fixture
    def pipeline(self, parser, detector):
        """Create integration pipeline"""
        return Pipeline(parser=parser, detector=detector)
    
    @pytest.fixture
    def sample_json_file(self):
        """Create temporary JSON file with sample data"""
        data = {
            "records": [
                {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0},
                {"feature1": 4.0, "feature2": 5.0, "feature3": 6.0},
                {"feature1": 100.0, "feature2": 200.0, "feature3": 300.0},  # anomaly
                {"feature1": 7.0, "feature2": 8.0, "feature3": 9.0}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_file = f.name
        
        yield temp_file
        os.unlink(temp_file)
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create temporary CSV file with sample data"""
        csv_data = """feature1,feature2,feature3
1.0,2.0,3.0
4.0,5.0,6.0
100.0,200.0,300.0
7.0,8.0,9.0"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_data)
            temp_file = f.name
        
        yield temp_file
        os.unlink(temp_file)
    
    def test_end_to_end_json_processing(self, pipeline, sample_json_file):
        """Test complete end-to-end processing with JSON file"""
        with patch.object(pipeline.detector, '_load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.2, 0.3, 0.9, 0.1])  # anomaly at index 2
            mock_load.return_value = mock_model
            
            result = pipeline.process_file(sample_json_file)
            
            assert isinstance(result, dict)
            assert 'detection_results' in result
            assert 'file_info' in result
            assert 'anomalies_detected' in result
            
            # Check that anomaly was detected
            assert result['anomalies_detected'] == 1
            assert result['detection_results'].predictions[2] == True
    
    def test_end_to_end_csv_processing(self, pipeline, sample_csv_file):
        """Test complete end-to-end processing with CSV file"""
        with patch.object(pipeline.detector, '_load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.1, 0.2, 0.8, 0.15])
            mock_load.return_value = mock_model
            
            result = pipeline.process_file(sample_csv_file)
            
            assert isinstance(result, dict)
            assert result['file_info']['format'] == 'csv'
            assert result['anomalies_detected'] == 1
    
    def test_parser_detector_data_flow(self, parser, detector, sample_json_file):
        """Test data flow between parser and detector"""
        # Parse the file
        parsed_data = parser.parse_file(sample_json_file)
        
        # Extract features for detection
        features = []
        for record in parsed_data['records']:
            features.append([record['feature1'], record['feature2'], record['feature3']])
        
        feature_array = np.array(features)
        
        # Run detection
        with patch.object(detector, '_load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.1, 0.2, 0.9, 0.1])
            mock_load.return_value = mock_model
            
            detection_result = detector.detect(feature_array)
            
            assert len(detection_result.predictions) == 4
            assert detection_result.predictions[2] == True  # anomaly detected
    
    def test_pipeline_error_handling(self, pipeline):
        """Test pipeline error handling"""
        # Test with non-existent file
        with pytest.raises(IntegrationError) as exc_info:
            pipeline.process_file("non_existent_file.json")
        
        assert "File processing failed" in str(exc_info.value)
    
    def test_pipeline_with_different_thresholds(self, pipeline, sample_json_file):
        """Test pipeline with different detection thresholds"""
        thresholds = [0.3, 0.5, 0.7, 0.9]
        results = []
        
        for threshold in thresholds:
            pipeline.detector.set_threshold(threshold)
            
            with patch.object(pipeline.detector, '_load_model') as mock_load:
                mock_model = Mock()
                mock_model.predict.return_value = np.array([0.2, 0.4, 0.6, 0.8])
                mock_load.return_value = mock_model
                
                result = pipeline.process_file(sample_json_file)
                results.append(result['anomalies_detected'])
        
        # Higher thresholds should detect fewer anomalies
        assert results[0] >= results[1] >= results[2] >= results[3]
    
    def test_batch_file_processing(self, pipeline, sample_json_file, sample_csv_file):
        """Test processing multiple files in batch"""
        files = [sample_json_file, sample_csv_file]
        
        with patch.object(pipeline.detector, '_load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.1, 0.2, 0.8, 0.1])
            mock_load.return_value = mock_model
            
            results = pipeline.process_batch(files)