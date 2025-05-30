import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from detector import Detector, DetectionError, DetectionResult


class TestDetector:
    """Test suite for Detector class"""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing"""
        return Detector(threshold=0.5, model_path="test_model.pkl")
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        return np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ])
    
    @pytest.fixture
    def anomaly_data(self):
        """Sample data with anomalies"""
        return np.array([
            [1.0, 2.0, 3.0],      # normal
            [4.0, 5.0, 6.0],      # normal
            [100.0, 200.0, 300.0], # anomaly
            [7.0, 8.0, 9.0]       # normal
        ])
    
    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector is not None
        assert detector.threshold == 0.5
        assert detector.model_path == "test_model.pkl"
        assert hasattr(detector, 'detect')
    
    def test_detector_initialization_default_params(self):
        """Test detector initialization with default parameters"""
        detector = Detector()
        assert detector.threshold == 0.7  # default threshold
        assert detector.model_path is None
    
    def test_detect_basic(self, detector, sample_data):
        """Test basic detection functionality"""
        with patch.object(detector, '_load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.3, 0.6, 0.2, 0.8])
            mock_load.return_value = mock_model
            
            result = detector.detect(sample_data)
            
            assert isinstance(result, DetectionResult)
            assert len(result.scores) == 4
            assert len(result.predictions) == 4
            assert result.predictions[1] == True   # 0.6 > 0.5
            assert result.predictions[3] == True   # 0.8 > 0.5
    
    def test_detect_empty_data(self, detector):
        """Test detection with empty data"""
        empty_data = np.array([]).reshape(0, 3)
        
        with pytest.raises(DetectionError) as exc_info:
            detector.detect(empty_data)
        
        assert "Empty data provided" in str(exc_info.value)
    
    def test_detect_invalid_shape(self, detector):
        """Test detection with invalid data shape"""
        invalid_data = np.array([1, 2, 3])  # 1D array instead of 2D
        
        with pytest.raises(DetectionError) as exc_info:
            detector.detect(invalid_data)
        
        assert "Invalid data shape" in str(exc_info.value)
    
    def test_detect_anomalies(self, detector, anomaly_data):
        """Test anomaly detection"""
        with patch.object(detector, '_load_model') as mock_load:
            mock_model = Mock()
            # High score for the anomaly row
            mock_model.predict.return_value = np.array([0.1, 0.2, 0.9, 0.15])
            mock_load.return_value = mock_model
            
            result = detector.detect(anomaly_data)
            
            assert result.predictions[2] == True  # anomaly detected
            assert result.predictions[0] == False  # normal data
            assert result.predictions[1] == False  # normal data
            assert result.predictions[3] == False  # normal data
    
    def test_set_threshold(self, detector):
        """Test threshold setting"""
        original_threshold = detector.threshold
        
        detector.set_threshold(0.8)
        assert detector.threshold == 0.8
        assert detector.threshold != original_threshold
    
    def test_invalid_threshold(self, detector):
        """Test setting invalid threshold"""
        with pytest.raises(ValueError) as exc_info:
            detector.set_threshold(-0.1)
        
        assert "Threshold must be between 0 and 1" in str(exc_info.value)
        
        with pytest.raises(ValueError):
            detector.set_threshold(1.5)
    
    @patch('pickle.load')
    @patch('builtins.open')
    def test_load_model_success(self, mock_open, mock_pickle_load, detector):
        """Test successful model loading"""
        mock_model = Mock()
        mock_pickle_load.return_value = mock_model
        
        result = detector._load_model()
        
        mock_open.assert_called_once_with("test_model.pkl", 'rb')
        assert result == mock_model
    
    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_load_model_file_not_found(self, mock_open, detector):
        """Test model loading with missing file"""
        with pytest.raises(DetectionError) as exc_info:
            detector._load_model()
        
        assert "Model file not found" in str(exc_info.value)
    
    def test_preprocess_data(self, detector, sample_data):
        """Test data preprocessing"""
        processed_data = detector._preprocess_data(sample_data)
        
        # Should normalize or scale the data
        assert processed_data.shape == sample_data.shape
        assert isinstance(processed_data, np.ndarray)
    
    def test_preprocess_data_with_scaling(self, detector):
        """Test data preprocessing with scaling"""
        data = np.array([[1, 100], [2, 200], [3, 300]])
        
        processed_data = detector._preprocess_data(data, scale=True)
        
        # After scaling, values should be normalized
        assert np.all(processed_data <= 1.0)
        assert np.all(processed_data >= -1.0) or np.all(processed_data >= 0.0)
    
    def test_batch_detection(self, detector):
        """Test batch detection on multiple datasets"""
        batch_data = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[9, 10], [11, 12]])
        ]
        
        with patch.object(detector, '_load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.side_effect = [
                np.array([0.3, 0.6]),
                np.array([0.4, 0.7]),
                np.array([0.2, 0.8])
            ]
            mock_load.return_value = mock_model
            
            results = detector.detect_batch(batch_data)
            
            assert len(results) == 3
            assert all(isinstance(r, DetectionResult) for r in results)
    
    def test_realtime_detection(self, detector):
        """Test real-time detection capability"""
        single_sample = np.array([[1.0, 2.0, 3.0]])
        
        with patch.object(detector, '_load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.6])
            mock_load.return_value = mock_model
            
            result = detector.detect_realtime(single_sample)
            
            assert isinstance(result, bool)
            assert result == True  # 0.6 > 0.5 threshold
    
    def test_detection_confidence(self, detector, sample_data):
        """Test detection confidence calculation"""
        with patch.object(detector, '_load_model') as mock_load:
            mock_model = Mock()
            scores = np.array([0.1, 0.6, 0.9, 0.3])
            mock_model.predict.return_value = scores
            mock_load.return_value = mock_model
            
            result = detector.detect(sample_data)
            
            # Test confidence calculation
            expected_confidence = np.abs(scores - detector.threshold)
            np.testing.assert_array_almost_equal(result.confidence, expected_confidence)
    
    def test_performance_metrics(self, detector):
        """Test performance metrics calculation"""
        y_true = np.array([False, True, True, False, True])
        y_pred = np.array([False, True, False, False, True])
        
        metrics = detector.calculate_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        
        # Check accuracy calculation
        expected_accuracy = 0.8  # 4 correct out of 5
        assert abs(metrics['accuracy'] - expected_accuracy) < 0.01
    
    def test_feature_importance(self, detector, sample_data):
        """Test feature importance extraction"""
        with patch.object(detector, '_load_model') as mock_load:
            mock_model = Mock()
            mock_model.feature_importances_ = np.array([0.3, 0.5, 0.2])
            mock_load.return_value = mock_model
            
            importance = detector.get_feature_importance()
            
            assert len(importance) == 3
            assert np.sum(importance) == 1.0  # Should sum to 1
    
    def test_model_validation(self, detector):
        """Test model validation"""
        with patch.object(detector, '_load_model') as mock_load:
            # Test with invalid model (no predict method)
            invalid_model = Mock(spec=[])  # No predict method
            mock_load.return_value = invalid_model
            
            with pytest.raises(DetectionError) as exc_info:
                detector._validate_model()
            
            assert "Invalid model" in str(exc_info.value)
    
    def test_detection_with_labels(self, detector, sample_data):
        """Test detection with known labels for evaluation"""
        true_labels = np.array([False, True, False, True])
        
        with patch.object(detector, '_load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.2, 0.7, 0.3, 0.8])
            mock_load.return_value = mock_model
            
            result = detector.detect(sample_data, labels=true_labels)
            
            assert hasattr(result, 'evaluation_metrics')
            assert 'accuracy' in result.evaluation_metrics
    
    def test_adaptive_threshold(self, detector, sample_data):
        """Test adaptive threshold adjustment"""
        with patch.object(detector, '_load_model') as mock_load:
            mock_model = Mock()
            scores = np.array([0.1, 0.3, 0.7, 0.9])
            mock_model.predict.return_value = scores
            mock_load.return_value = mock_model
            
            # Enable adaptive threshold
            detector.enable_adaptive_threshold(target_fpr=0.1)
            result = detector.detect(sample_data)
            
            # Threshold should have been adjusted
            assert detector.threshold != 0.5
    
    @pytest.mark.parametrize("threshold,expected_detections", [
        (0.3, [False, True, True, True]),
        (0.6, [False, False, True, True]),
        (0.9, [False, False, False, True]),
    ])
    def test_threshold_sensitivity(self, detector, sample_data, threshold, expected_detections):
        """Test detection sensitivity to threshold changes"""
        detector.set_threshold(threshold)
        
        with patch.object(detector, '_load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.2, 0.5, 0.7, 0.95])
            mock_load.return_value = mock_model
            
            result = detector.detect(sample_data)
            
            assert list(result.predictions) == expected_detections
    
    def test_concurrent_detection(self, detector):
        """Test concurrent detection capability"""
        import threading
        
        data_batches = [np.random.rand(10, 3) for _ in range(5)]
        results = []
        
        def detect_worker(data):
            with patch.object(detector, '_load_model') as mock_load:
                mock_model = Mock()
                mock_model.predict.return_value = np.random.rand(len(data))
                mock_load.return_value = mock_model
                
                result = detector.detect(data)
                results.append(result)
        
        threads = []
        for data in data_batches:
            thread = threading.Thread(target=detect_worker, args=(data,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 5
    
    def test_memory_cleanup(self, detector):
        """Test memory cleanup after detection"""
        large_data = np.random.rand(1000, 10)
        
        with patch.object(detector, '_load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.random.rand(1000)
            mock_load.return_value = mock_model
            
            result = detector.detect(large_data)
            
            # Test that detector doesn't hold references to large data
            assert not hasattr(detector, '_cached_data')
            assert isinstance(result, DetectionResult)