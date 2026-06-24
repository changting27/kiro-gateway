# -*- coding: utf-8 -*-

"""
Unit-тесты для DebugLogger.
Проверяет логику буферизации и записи debug логов в разных режимах.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestDebugLoggerModeOff:
    """Тесты для режима DEBUG_MODE=off."""
    
    def test_prepare_new_request_does_nothing(self, tmp_path):
        """
        Что он делает: Проверяет, что prepare_new_request ничего не делает в режиме off.
        Цель: Убедиться, что в режиме off директория не создаётся.
        """
        print("Настройка: Режим off...")
        with patch('kiro.debug_logger.DEBUG_MODE', 'off'):
            with patch('kiro.debug_logger.DEBUG_DIR', str(tmp_path / "debug_logs")):
                # Пересоздаём экземпляр с новыми настройками
                from kiro.debug_logger import DebugLogger
                logger = DebugLogger.__new__(DebugLogger)
                logger._initialized = False
                logger.__init__()
                logger.debug_dir = tmp_path / "debug_logs"
                
                print("Действие: Вызов prepare_new_request...")
                logger.prepare_new_request()
                
                print(f"Проверяем, что директория не создана...")
                assert not (tmp_path / "debug_logs").exists()
    
    def test_log_request_body_does_nothing(self, tmp_path):
        """
        Что он делает: Проверяет, что log_request_body ничего не делает в режиме off.
        Цель: Убедиться, что данные не записываются.
        """
        print("Настройка: Режим off...")
        with patch('kiro.debug_logger.DEBUG_MODE', 'off'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = tmp_path / "debug_logs"
            
            print("Действие: Вызов log_request_body...")
            logger.log_request_body(b'{"test": "data"}')
            
            print(f"Проверяем, что файл не создан...")
            assert not (tmp_path / "debug_logs" / "request_body.json").exists()


class TestDebugLoggerModeAll:
    """Тесты для режима DEBUG_MODE=all."""
    
    def test_prepare_new_request_clears_directory(self, tmp_path):
        """
        Что он делает: Проверяет, что prepare_new_request очищает директорию в режиме all.
        Цель: Убедиться, что старые логи удаляются.
        """
        print("Настройка: Режим all, создаём старый файл...")
        debug_dir = tmp_path / "debug_logs"
        debug_dir.mkdir()
        old_file = debug_dir / "old_file.txt"
        old_file.write_text("old content")
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir
            
            print("Действие: Вызов prepare_new_request...")
            logger.prepare_new_request()
            
            print(f"Проверяем, что старый файл удалён...")
            assert not old_file.exists()
            print(f"Проверяем, что директория существует...")
            assert debug_dir.exists()
    
    def test_log_request_body_writes_immediately(self, tmp_path):
        """
        Что он делает: Проверяет, что log_request_body пишет сразу в файл в режиме all.
        Цель: Убедиться, что данные записываются немедленно.
        """
        print("Настройка: Режим all...")
        debug_dir = tmp_path / "debug_logs"
        debug_dir.mkdir()
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir
            
            print("Действие: Вызов log_request_body...")
            test_data = b'{"model": "test", "messages": []}'
            logger.log_request_body(test_data)
            
            print(f"Проверяем, что файл создан...")
            file_path = debug_dir / "request_body.json"
            assert file_path.exists()
            
            print(f"Проверяем содержимое файла...")
            content = json.loads(file_path.read_text())
            assert content["model"] == "test"
    
    def test_log_kiro_request_body_writes_immediately(self, tmp_path):
        """
        Что он делает: Проверяет, что log_kiro_request_body пишет сразу в файл в режиме all.
        Цель: Убедиться, что Kiro payload записывается немедленно.
        """
        print("Настройка: Режим all...")
        debug_dir = tmp_path / "debug_logs"
        debug_dir.mkdir()
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir
            
            print("Действие: Вызов log_kiro_request_body...")
            test_data = b'{"conversationState": {}}'
            logger.log_kiro_request_body(test_data)
            
            print(f"Проверяем, что файл создан...")
            file_path = debug_dir / "kiro_request_body.json"
            assert file_path.exists()
    
    def test_log_raw_chunk_appends_to_file(self, tmp_path):
        """
        Что он делает: Проверяет, что log_raw_chunk дописывает в файл в режиме all.
        Цель: Убедиться, что чанки накапливаются.
        """
        print("Настройка: Режим all...")
        debug_dir = tmp_path / "debug_logs"
        debug_dir.mkdir()
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir
            
            print("Действие: Вызов log_raw_chunk дважды...")
            logger.log_raw_chunk(b'chunk1')
            logger.log_raw_chunk(b'chunk2')
            
            print(f"Проверяем содержимое файла...")
            file_path = debug_dir / "response_stream_raw.txt"
            content = file_path.read_bytes()
            assert content == b'chunk1chunk2'


class TestDebugLoggerModeErrors:
    """Тесты для режима DEBUG_MODE=errors."""
    
    def test_log_request_body_buffers_data(self, tmp_path):
        """
        Что он делает: Проверяет, что log_request_body буферизует данные в режиме errors.
        Цель: Убедиться, что данные не записываются сразу.
        """
        print("Настройка: Режим errors...")
        debug_dir = tmp_path / "debug_logs"
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir
            
            print("Действие: Вызов log_request_body...")
            test_data = b'{"test": "buffered"}'
            logger.log_request_body(test_data)
            
            print(f"Проверяем, что файл НЕ создан...")
            assert not debug_dir.exists()
            
            print(f"Проверяем, что данные в буфере...")
            assert logger._request_body_buffer == test_data
    
    def test_flush_on_error_writes_buffers(self, tmp_path):
        """
        Что он делает: Проверяет, что flush_on_error записывает буферы в файлы.
        Цель: Убедиться, что при ошибке данные сохраняются.
        """
        print("Настройка: Режим errors, заполняем буферы...")
        debug_dir = tmp_path / "debug_logs"
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir
            
            # Заполняем буферы
            logger.log_request_body(b'{"request": "body"}')
            logger.log_kiro_request_body(b'{"kiro": "request"}')
            logger.log_raw_chunk(b'raw_chunk')
            logger.log_modified_chunk(b'modified_chunk')
            
            print("Действие: Вызов flush_on_error...")
            logger.flush_on_error(400, "Bad Request")
            
            print(f"Проверяем, что все файлы созданы...")
            assert (debug_dir / "request_body.json").exists()
            assert (debug_dir / "kiro_request_body.json").exists()
            assert (debug_dir / "response_stream_raw.txt").exists()
            assert (debug_dir / "response_stream_modified.txt").exists()
            assert (debug_dir / "error_info.json").exists()
            
            print(f"Проверяем error_info.json...")
            error_info = json.loads((debug_dir / "error_info.json").read_text())
            assert error_info["status_code"] == 400
            assert error_info["error_message"] == "Bad Request"
    
    def test_flush_on_error_clears_buffers(self, tmp_path):
        """
        Что он делает: Проверяет, что flush_on_error очищает буферы после записи.
        Цель: Убедиться, что буферы не накапливаются между запросами.
        """
        print("Настройка: Режим errors...")
        debug_dir = tmp_path / "debug_logs"
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir
            
            logger.log_request_body(b'{"test": "data"}')
            
            print("Действие: Вызов flush_on_error...")
            logger.flush_on_error(500, "Error")
            
            print(f"Проверяем, что буферы очищены...")
            assert logger._request_body_buffer is None
            assert logger._kiro_request_body_buffer is None
            assert len(logger._raw_chunks_buffer) == 0
            assert len(logger._modified_chunks_buffer) == 0
    
    def test_discard_buffers_clears_without_writing(self, tmp_path):
        """
        Что он делает: Проверяет, что discard_buffers очищает буферы без записи.
        Цель: Убедиться, что успешные запросы не оставляют логов.
        """
        print("Настройка: Режим errors, заполняем буферы...")
        debug_dir = tmp_path / "debug_logs"
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir
            
            logger.log_request_body(b'{"test": "data"}')
            logger.log_raw_chunk(b'chunk')
            
            print("Действие: Вызов discard_buffers...")
            logger.discard_buffers()
            
            print(f"Проверяем, что директория НЕ создана...")
            assert not debug_dir.exists()
            
            print(f"Проверяем, что буферы очищены...")
            assert logger._request_body_buffer is None
            assert len(logger._raw_chunks_buffer) == 0
    
    def test_flush_on_error_writes_error_info_in_mode_all(self, tmp_path):
        """
        Что он делает: Проверяет, что flush_on_error записывает error_info.json в режиме all.
        Цель: Убедиться, что информация об ошибке сохраняется в обоих режимах.
        """
        print("Настройка: Режим all...")
        debug_dir = tmp_path / "debug_logs"
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir
            
            print("Действие: Вызов flush_on_error...")
            logger.flush_on_error(400, "Bad Request")
            
            print(f"Проверяем, что error_info.json создан...")
            assert (debug_dir / "error_info.json").exists()
            
            print(f"Проверяем содержимое error_info.json...")
            error_info = json.loads((debug_dir / "error_info.json").read_text())
            assert error_info["status_code"] == 400
            assert error_info["error_message"] == "Bad Request"


class TestDebugLoggerLogErrorInfo:
    """Тесты для метода log_error_info()."""
    
    def test_log_error_info_writes_in_mode_all(self, tmp_path):
        """
        Что он делает: Проверяет, что log_error_info записывает файл в режиме all.
        Цель: Убедиться, что error_info.json создаётся при ошибках.
        """
        print("Настройка: Режим all...")
        debug_dir = tmp_path / "debug_logs"
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir
            
            print("Действие: Вызов log_error_info...")
            logger.log_error_info(500, "Internal Server Error")
            
            print(f"Проверяем, что error_info.json создан...")
            error_file = debug_dir / "error_info.json"
            assert error_file.exists()
            
            print(f"Проверяем содержимое...")
            error_info = json.loads(error_file.read_text())
            assert error_info["status_code"] == 500
            assert error_info["error_message"] == "Internal Server Error"
    
    def test_log_error_info_writes_in_mode_errors(self, tmp_path):
        """
        Что он делает: Проверяет, что log_error_info записывает файл в режиме errors.
        Цель: Убедиться, что метод работает в обоих режимах.
        """
        print("Настройка: Режим errors...")
        debug_dir = tmp_path / "debug_logs"
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir
            
            print("Действие: Вызов log_error_info...")
            logger.log_error_info(404, "Not Found")
            
            print(f"Проверяем, что error_info.json создан...")
            error_file = debug_dir / "error_info.json"
            assert error_file.exists()
    
    def test_log_error_info_does_nothing_in_mode_off(self, tmp_path):
        """
        Что он делает: Проверяет, что log_error_info ничего не делает в режиме off.
        Цель: Убедиться, что в режиме off файлы не создаются.
        """
        print("Настройка: Режим off...")
        debug_dir = tmp_path / "debug_logs"
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'off'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir
            
            print("Действие: Вызов log_error_info...")
            logger.log_error_info(500, "Error")
            
            print(f"Проверяем, что директория НЕ создана...")
            assert not debug_dir.exists()


class TestDebugLoggerHelperMethods:
    """Тесты для вспомогательных методов DebugLogger."""
    
    def test_is_enabled_returns_true_for_errors(self):
        """
        Что он делает: Проверяет _is_enabled() для режима errors.
        Цель: Убедиться, что режим errors считается включённым.
        """
        print("Настройка: Режим errors...")
        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            
            print(f"Проверяем _is_enabled()...")
            assert logger._is_enabled() is True
    
    def test_is_enabled_returns_true_for_all(self):
        """
        Что он делает: Проверяет _is_enabled() для режима all.
        Цель: Убедиться, что режим all считается включённым.
        """
        print("Настройка: Режим all...")
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            
            print(f"Проверяем _is_enabled()...")
            assert logger._is_enabled() is True
    
    def test_is_enabled_returns_false_for_off(self):
        """
        Что он делает: Проверяет _is_enabled() для режима off.
        Цель: Убедиться, что режим off считается выключенным.
        """
        print("Настройка: Режим off...")
        with patch('kiro.debug_logger.DEBUG_MODE', 'off'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            
            print(f"Проверяем _is_enabled()...")
            assert logger._is_enabled() is False
    
    def test_is_immediate_write_returns_true_for_all(self):
        """
        Что он делает: Проверяет _is_immediate_write() для режима all.
        Цель: Убедиться, что режим all пишет сразу.
        """
        print("Настройка: Режим all...")
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            
            print(f"Проверяем _is_immediate_write()...")
            assert logger._is_immediate_write() is True
    
    def test_is_immediate_write_returns_false_for_errors(self):
        """
        Что он делает: Проверяет _is_immediate_write() для режима errors.
        Цель: Убедиться, что режим errors буферизует.
        """
        print("Настройка: Режим errors...")
        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            
            print(f"Проверяем _is_immediate_write()...")
            assert logger._is_immediate_write() is False


class TestDebugLoggerJsonHandling:
    """Тесты для обработки JSON в DebugLogger."""
    
    def test_log_request_body_formats_json_pretty(self, tmp_path):
        """
        Что он делает: Проверяет, что JSON форматируется красиво.
        Цель: Убедиться, что JSON читаем в файле.
        """
        print("Настройка: Режим all...")
        debug_dir = tmp_path / "debug_logs"
        debug_dir.mkdir()
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir
            
            print("Действие: Вызов log_request_body с JSON...")
            logger.log_request_body(b'{"key":"value"}')
            
            print(f"Проверяем форматирование...")
            content = (debug_dir / "request_body.json").read_text()
            # Должен быть отформатирован с отступами
            assert "  " in content or "\n" in content
    
    def test_log_request_body_handles_invalid_json(self, tmp_path):
        """
        Что он делает: Проверяет обработку невалидного JSON.
        Цель: Убедиться, что невалидный JSON записывается как есть.
        """
        print("Настройка: Режим all...")
        debug_dir = tmp_path / "debug_logs"
        debug_dir.mkdir()
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            logger = DebugLogger.__new__(DebugLogger)
            logger._initialized = False
            logger.__init__()
            logger.debug_dir = debug_dir
            
            print("Действие: Вызов log_request_body с невалидным JSON...")
            invalid_data = b'not a json {{'
            logger.log_request_body(invalid_data)
            
            print(f"Проверяем, что данные записаны как есть...")
            content = (debug_dir / "request_body.json").read_bytes()
            assert content == invalid_data


class TestDebugLoggerAppLogsCapture:
    """Тесты для захвата логов приложения (app_logs.txt)."""
    
    def test_prepare_new_request_sets_up_log_capture(self, tmp_path):
        """
        Что он делает: Проверяет, что prepare_new_request настраивает захват логов.
        Цель: Убедиться, что sink для логов создаётся.
        """
        print("Настройка: Режим all...")
        debug_dir = tmp_path / "debug_logs"
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            dbg_logger = DebugLogger.__new__(DebugLogger)
            dbg_logger._initialized = False
            dbg_logger.__init__()
            dbg_logger.debug_dir = debug_dir
            
            print("Действие: Вызов prepare_new_request...")
            dbg_logger.prepare_new_request()
            
            print(f"Проверяем, что sink создан...")
            assert dbg_logger._loguru_sink_id is not None
            
            # Очистка
            dbg_logger._clear_app_logs_buffer()
    
    def test_flush_on_error_writes_app_logs_in_mode_errors(self, tmp_path):
        """
        Что он делает: Проверяет, что flush_on_error записывает app_logs.txt в режиме errors.
        Цель: Убедиться, что логи приложения сохраняются при ошибках.
        """
        print("Настройка: Режим errors...")
        debug_dir = tmp_path / "debug_logs"
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger
            from loguru import logger as loguru_logger
            
            dbg_logger = DebugLogger.__new__(DebugLogger)
            dbg_logger._initialized = False
            dbg_logger.__init__()
            dbg_logger.debug_dir = debug_dir
            
            # Настраиваем захват логов
            dbg_logger.prepare_new_request()
            
            # Добавляем данные в буфер чтобы flush сработал
            dbg_logger.log_request_body(b'{"test": "data"}')
            
            # Пишем тестовый лог напрямую в буфер (имитация)
            dbg_logger._app_logs_buffer.write("Test log message\n")
            
            print("Действие: Вызов flush_on_error...")
            dbg_logger.flush_on_error(500, "Test Error")
            
            print(f"Проверяем, что app_logs.txt создан...")
            app_logs_file = debug_dir / "app_logs.txt"
            assert app_logs_file.exists()
            
            print(f"Проверяем содержимое...")
            content = app_logs_file.read_text()
            assert "Test log message" in content
    
    def test_discard_buffers_saves_logs_in_mode_all(self, tmp_path):
        """
        Что он делает: Проверяет, что discard_buffers сохраняет логи в режиме all.
        Цель: Убедиться, что даже успешные запросы сохраняют логи в режиме all.
        """
        print("Настройка: Режим all...")
        debug_dir = tmp_path / "debug_logs"
        debug_dir.mkdir()
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            
            dbg_logger = DebugLogger.__new__(DebugLogger)
            dbg_logger._initialized = False
            dbg_logger.__init__()
            dbg_logger.debug_dir = debug_dir
            
            # Настраиваем захват логов
            dbg_logger.prepare_new_request()
            
            # Пишем тестовый лог напрямую в буфер
            dbg_logger._app_logs_buffer.write("Success log message\n")
            
            print("Действие: Вызов discard_buffers...")
            dbg_logger.discard_buffers()
            
            print(f"Проверяем, что app_logs.txt создан...")
            app_logs_file = debug_dir / "app_logs.txt"
            assert app_logs_file.exists()
            
            print(f"Проверяем содержимое...")
            content = app_logs_file.read_text()
            assert "Success log message" in content
    
    def test_discard_buffers_does_not_save_logs_in_mode_errors(self, tmp_path):
        """
        Что он делает: Проверяет, что discard_buffers НЕ сохраняет логи в режиме errors.
        Цель: Убедиться, что успешные запросы не оставляют логов в режиме errors.
        """
        print("Настройка: Режим errors...")
        debug_dir = tmp_path / "debug_logs"
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            from kiro.debug_logger import DebugLogger
            
            dbg_logger = DebugLogger.__new__(DebugLogger)
            dbg_logger._initialized = False
            dbg_logger.__init__()
            dbg_logger.debug_dir = debug_dir
            
            # Настраиваем захват логов
            dbg_logger.prepare_new_request()
            
            # Пишем тестовый лог напрямую в буфер
            dbg_logger._app_logs_buffer.write("Should not be saved\n")
            
            print("Действие: Вызов discard_buffers...")
            dbg_logger.discard_buffers()
            
            print(f"Проверяем, что директория НЕ создана...")
            assert not debug_dir.exists()
    
    def test_clear_app_logs_buffer_removes_sink(self, tmp_path):
        """
        Что он делает: Проверяет, что _clear_app_logs_buffer удаляет sink.
        Цель: Убедиться, что sink корректно удаляется.
        """
        print("Настройка: Режим all...")
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            
            dbg_logger = DebugLogger.__new__(DebugLogger)
            dbg_logger._initialized = False
            dbg_logger.__init__()
            dbg_logger.debug_dir = tmp_path / "debug_logs"
            
            # Настраиваем захват логов
            dbg_logger.prepare_new_request()
            sink_id = dbg_logger._loguru_sink_id
            assert sink_id is not None
            
            print("Действие: Вызов _clear_app_logs_buffer...")
            dbg_logger._clear_app_logs_buffer()
            
            print(f"Проверяем, что sink_id сброшен...")
            assert dbg_logger._loguru_sink_id is None
    
    def test_app_logs_not_saved_when_empty(self, tmp_path):
        """
        Что он делает: Проверяет, что пустые логи не создают файл.
        Цель: Убедиться, что app_logs.txt не создаётся если логов нет.
        """
        print("Настройка: Режим all...")
        debug_dir = tmp_path / "debug_logs"
        debug_dir.mkdir()
        
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            from kiro.debug_logger import DebugLogger
            
            dbg_logger = DebugLogger.__new__(DebugLogger)
            dbg_logger._initialized = False
            dbg_logger.__init__()
            dbg_logger.debug_dir = debug_dir
            
            # НЕ пишем ничего в буфер
            
            print("Действие: Вызов _write_app_logs_to_file...")
            dbg_logger._write_app_logs_to_file()
            
            print(f"Проверяем, что app_logs.txt НЕ создан...")
            app_logs_file = debug_dir / "app_logs.txt"
            assert not app_logs_file.exists()


def _make_logger(debug_dir):
    """
    Build a fresh DebugLogger instance bound to a temp directory.

    The logger is a singleton, so tests recreate the instance to avoid state
    bleed between cases (matches the pattern used throughout this file).

    Args:
        debug_dir: Directory the logger should write artifacts into.

    Returns:
        A freshly-initialized DebugLogger pointed at debug_dir.
    """
    from kiro.debug_logger import DebugLogger
    dbg_logger = DebugLogger.__new__(DebugLogger)
    dbg_logger._initialized = False
    dbg_logger.__init__()
    dbg_logger.debug_dir = debug_dir
    return dbg_logger


class TestDebugLoggerRequestFlush:
    """
    Tests for request_flush() forcing buffered artifacts to be persisted on an
    otherwise-successful (HTTP 200) request - the Kiro tool-call truncation case.
    """

    def test_request_flush_then_discard_writes_buffers_in_errors_mode(self, tmp_path):
        """
        What it does: After request_flush(), discard_buffers() flushes the raw
            stream and request bodies to disk in "errors" mode.
        Purpose: Tool-call truncation returns HTTP 200, so the success path
            (discard_buffers) must persist evidence instead of dropping it.
        """
        debug_dir = tmp_path / "debug_logs"
        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            # Arrange
            dbg_logger = _make_logger(debug_dir)
            dbg_logger.log_request_body(b'{"model": "claude-opus-4.6"}')
            dbg_logger.log_raw_chunk(b'{"toolUseId":"x","name":"Write"}')

            # Act
            dbg_logger.request_flush(
                reason="Tool call truncated by Kiro API: tool='Write', size=64 bytes",
                status_code=200,
            )
            dbg_logger.discard_buffers()

            # Assert
            assert (debug_dir / "response_stream_raw.txt").exists()
            assert (debug_dir / "request_body.json").exists()
            assert (debug_dir / "error_info.json").exists()
            error_info = json.loads((debug_dir / "error_info.json").read_text())
            assert error_info["status_code"] == 200
            assert "truncated" in error_info["error_message"].lower()

    def test_request_flush_records_error_info_in_all_mode(self, tmp_path):
        """
        What it does: In "all" mode, request_flush()+discard_buffers() records
            error_info.json (data is already written immediately in all mode).
        Purpose: Verify the feature behaves consistently in both enabled modes.
        """
        debug_dir = tmp_path / "debug_logs"
        with patch('kiro.debug_logger.DEBUG_MODE', 'all'):
            # Arrange
            dbg_logger = _make_logger(debug_dir)

            # Act
            dbg_logger.request_flush(reason="Tool call truncated by Kiro API", status_code=200)
            dbg_logger.discard_buffers()

            # Assert
            assert (debug_dir / "error_info.json").exists()
            error_info = json.loads((debug_dir / "error_info.json").read_text())
            assert error_info["status_code"] == 200

    def test_request_flush_is_noop_in_off_mode(self, tmp_path):
        """
        What it does: request_flush() does nothing when DEBUG_MODE=off, and
            discard_buffers() writes no files.
        Purpose: Disabled logging must never touch the filesystem.
        """
        debug_dir = tmp_path / "debug_logs"
        with patch('kiro.debug_logger.DEBUG_MODE', 'off'):
            # Arrange
            dbg_logger = _make_logger(debug_dir)
            dbg_logger.log_raw_chunk(b'chunk')

            # Act
            dbg_logger.request_flush(reason="truncation", status_code=200)
            dbg_logger.discard_buffers()

            # Assert
            assert dbg_logger._flush_requested is False
            assert not debug_dir.exists()

    def test_discard_without_request_flush_does_not_write(self, tmp_path):
        """
        What it does: Without request_flush(), discard_buffers() keeps the
            original "drop on success" behaviour in "errors" mode.
        Purpose: Regression guard - normal successful requests must not start
            leaving debug artifacts behind.
        """
        debug_dir = tmp_path / "debug_logs"
        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            # Arrange
            dbg_logger = _make_logger(debug_dir)
            dbg_logger.log_raw_chunk(b'chunk')

            # Act
            dbg_logger.discard_buffers()

            # Assert
            assert not debug_dir.exists()
            assert len(dbg_logger._raw_chunks_buffer) == 0

    def test_request_flush_is_idempotent_first_reason_wins(self, tmp_path):
        """
        What it does: Multiple request_flush() calls keep the first reason and
            status (e.g. several truncated tools in one response).
        Purpose: The earliest detected cause is the most useful; later calls
            must not overwrite it.
        """
        debug_dir = tmp_path / "debug_logs"
        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            # Arrange
            dbg_logger = _make_logger(debug_dir)
            dbg_logger.log_raw_chunk(b'chunk')

            # Act
            dbg_logger.request_flush(reason="first cause", status_code=200)
            dbg_logger.request_flush(reason="second cause", status_code=500)
            dbg_logger.discard_buffers()

            # Assert
            error_info = json.loads((debug_dir / "error_info.json").read_text())
            assert error_info["status_code"] == 200
            assert error_info["error_message"] == "first cause"

    def test_prepare_new_request_resets_flush_flag(self, tmp_path):
        """
        What it does: prepare_new_request() clears a leftover flush request so a
            truncation in request N never forces a flush in request N+1.
        Purpose: Prevent cross-request state bleed on the shared singleton.
        """
        debug_dir = tmp_path / "debug_logs"
        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            # Arrange
            dbg_logger = _make_logger(debug_dir)
            dbg_logger.request_flush(reason="stale truncation", status_code=200)
            assert dbg_logger._flush_requested is True

            # Act
            dbg_logger.prepare_new_request()
            dbg_logger.log_raw_chunk(b'fresh chunk')
            dbg_logger.discard_buffers()

            # Assert
            assert dbg_logger._flush_requested is False
            assert not debug_dir.exists()

    def test_real_error_still_flushes_when_flag_set(self, tmp_path):
        """
        What it does: A genuine flush_on_error() (e.g. HTTP 500) still works when
            a truncation flush was also requested, and the explicit error status
            is preserved.
        Purpose: A real upstream error must not be masked by the truncation flag.
        """
        debug_dir = tmp_path / "debug_logs"
        with patch('kiro.debug_logger.DEBUG_MODE', 'errors'):
            # Arrange
            dbg_logger = _make_logger(debug_dir)
            dbg_logger.log_raw_chunk(b'partial chunk')
            dbg_logger.request_flush(reason="truncation", status_code=200)

            # Act - route hits a real error and calls flush_on_error directly
            dbg_logger.flush_on_error(500, "Upstream failure")

            # Assert
            error_info = json.loads((debug_dir / "error_info.json").read_text())
            assert error_info["status_code"] == 500
            assert error_info["error_message"] == "Upstream failure"
            assert dbg_logger._flush_requested is False
