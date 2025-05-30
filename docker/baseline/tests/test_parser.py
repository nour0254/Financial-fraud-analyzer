import pytest
import json
from unittest.mock import Mock, patch, mock_open
from parser import Parser, ParseError


class TestParser:
    """Test suite for Parser class"""
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance for testing"""
        return Parser()
    
    @pytest.fixture
    def sample_json_data(self):
        """Sample JSON data for testing"""
        return {
            "name": "test_item",
            "value": 42,
            "items": ["item1", "item2", "item3"],
            "metadata": {
                "created": "2024-01-01",
                "version": "1.0"
            }
        }
    
    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV data for testing"""
        return "name,age,city\nJohn,25,New York\nJane,30,London\nBob,35,Paris"
    
    def test_parser_initialization(self, parser):
        """Test parser initialization"""
        assert parser is not None
        assert hasattr(parser, 'parse')
    
    def test_parse_json_string(self, parser, sample_json_data):
        """Test parsing valid JSON string"""
        json_string = json.dumps(sample_json_data)
        result = parser.parse_json(json_string)
        
        assert result == sample_json_data
        assert result["name"] == "test_item"
        assert result["value"] == 42
        assert len(result["items"]) == 3
    
    def test_parse_invalid_json(self, parser):
        """Test parsing invalid JSON string"""
        invalid_json = '{"name": "test", "value": }'
        
        with pytest.raises(ParseError) as exc_info:
            parser.parse_json(invalid_json)
        
        assert "Invalid JSON format" in str(exc_info.value)
    
    def test_parse_empty_json(self, parser):
        """Test parsing empty JSON string"""
        empty_json = ""
        
        with pytest.raises(ParseError):
            parser.parse_json(empty_json)
    
    def test_parse_csv_string(self, parser, sample_csv_data):
        """Test parsing CSV string"""
        result = parser.parse_csv(sample_csv_data)
        
        assert len(result) == 3
        assert result[0]["name"] == "John"
        assert result[0]["age"] == "25"
        assert result[1]["city"] == "London"
    
    def test_parse_csv_with_custom_delimiter(self, parser):
        """Test parsing CSV with custom delimiter"""
        csv_data = "name;age;city\nJohn;25;New York"
        result = parser.parse_csv(csv_data, delimiter=';')
        
        assert len(result) == 1
        assert result[0]["name"] == "John"
        assert result[0]["age"] == "25"
    
    def test_parse_empty_csv(self, parser):
        """Test parsing empty CSV"""
        empty_csv = ""
        result = parser.parse_csv(empty_csv)
        
        assert result == []
    
    def test_parse_csv_headers_only(self, parser):
        """Test parsing CSV with headers only"""
        headers_only = "name,age,city"
        result = parser.parse_csv(headers_only)
        
        assert result == []
    
    @patch("builtins.open", new_callable=mock_open)
    def test_parse_file_json(self, mock_file, parser, sample_json_data):
        """Test parsing JSON file"""
        json_content = json.dumps(sample_json_data)
        mock_file.return_value.read.return_value = json_content
        
        result = parser.parse_file("test.json")
        
        mock_file.assert_called_once_with("test.json", 'r', encoding='utf-8')
        assert result == sample_json_data
    
    @patch("builtins.open", new_callable=mock_open)
    def test_parse_file_csv(self, mock_file, parser, sample_csv_data):
        """Test parsing CSV file"""
        mock_file.return_value.read.return_value = sample_csv_data
        
        result = parser.parse_file("test.csv")
        
        mock_file.assert_called_once_with("test.csv", 'r', encoding='utf-8')
        assert len(result) == 3
        assert result[0]["name"] == "John"
    
    def test_parse_file_unsupported_format(self, parser):
        """Test parsing file with unsupported format"""
        with pytest.raises(ParseError) as exc_info:
            parser.parse_file("test.txt")
        
        assert "Unsupported file format" in str(exc_info.value)
    
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_parse_file_not_found(self, mock_file, parser):
        """Test parsing non-existent file"""
        with pytest.raises(ParseError) as exc_info:
            parser.parse_file("nonexistent.json")
        
        assert "File not found" in str(exc_info.value)
    
    def test_validate_json_schema(self, parser):
        """Test JSON schema validation"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name"]
        }
        
        valid_data = {"name": "John", "age": 25}
        invalid_data = {"age": 25}  # missing required 'name'
        
        assert parser.validate_schema(valid_data, schema) is True
        
        with pytest.raises(ParseError):
            parser.validate_schema(invalid_data, schema)
    
    def test_parse_xml_basic(self, parser):
        """Test basic XML parsing"""
        xml_data = """<?xml version="1.0"?>
        <root>
            <item>
                <name>test</name>
                <value>42</value>
            </item>
        </root>"""
        
        result = parser.parse_xml(xml_data)
        assert result is not None
        assert "root" in result
    
    def test_parse_malformed_xml(self, parser):
        """Test parsing malformed XML"""
        malformed_xml = "<root><item><name>test</item></root>"
        
        with pytest.raises(ParseError):
            parser.parse_xml(malformed_xml)
    
    def test_parse_with_encoding(self, parser):
        """Test parsing with different encodings"""
        data = "name,value\ntest,42"
        
        # Test UTF-8 encoding
        result = parser.parse_csv(data, encoding='utf-8')
        assert len(result) == 1
        assert result[0]["name"] == "test"
    
    def test_parse_large_file_streaming(self, parser):
        """Test streaming parse for large files"""
        # Mock a large CSV file
        large_csv = "name,value\n" + "\n".join([f"item{i},{i}" for i in range(1000)])
        
        with patch("builtins.open", mock_open(read_data=large_csv)):
            result = parser.parse_file_streaming("large.csv", chunk_size=100)
            
            # Should return a generator
            assert hasattr(result, '__iter__')
    
    def test_error_handling_graceful_degradation(self, parser):
        """Test graceful error handling"""
        # Test with partially corrupted data
        partial_json = '{"valid": "data", "invalid": }'
        
        with pytest.raises(ParseError) as exc_info:
            parser.parse_json(partial_json)
        
        # Should provide helpful error message
        assert "line" in str(exc_info.value).lower() or "position" in str(exc_info.value).lower()
    
    def test_parse_nested_structures(self, parser):
        """Test parsing deeply nested structures"""
        nested_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "data": ["item1", "item2"]
                    }
                }
            }
        }
        
        json_string = json.dumps(nested_data)
        result = parser.parse_json(json_string)
        
        assert result["level1"]["level2"]["level3"]["data"][0] == "item1"
    
    @pytest.mark.parametrize("file_ext,content,expected_type", [
        (".json", '{"key": "value"}', dict),
        (".csv", "name,value\ntest,42", list),
    ])
    def test_parse_file_types(self, parser, file_ext, content, expected_type):
        """Test parsing different file types"""
        with patch("builtins.open", mock_open(read_data=content)):
            result = parser.parse_file(f"test{file_ext}")
            assert isinstance(result, expected_type)
    
    def test_memory_efficiency(self, parser):
        """Test memory-efficient parsing for large data"""
        # This test would typically involve monitoring memory usage
        # For now, we'll test that the parser can handle reasonably large data
        large_json = json.dumps({"items": list(range(10000))})
        
        result = parser.parse_json(large_json)
        assert len(result["items"]) == 10000
        assert result["items"][0] == 0
        assert result["items"][-1] == 9999