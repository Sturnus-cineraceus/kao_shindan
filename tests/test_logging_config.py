import logging

from app.logging_config import configure_logging


def test_configure_logging_writes_console_and_file(tmp_path, monkeypatch, capsys):
    target = tmp_path / 'app.log'
    monkeypatch.setenv('LOG_FILE', str(target))
    monkeypatch.setenv('LOG_LEVEL', 'DEBUG')
    monkeypatch.setenv('LOG_MAX_BYTES', '512')
    monkeypatch.setenv('LOG_BACKUP_COUNT', '2')
    logger = configure_logging()
    before = len(logging.getLogger().handlers)
    configure_logging()
    assert len(logging.getLogger().handlers) == before
    logger.debug('safe debug event')
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert 'safe debug event' in target.read_text(encoding='utf-8')
    assert 'safe debug event' in capsys.readouterr().err
    assert any(getattr(handler, '_kao_handler', False) and handler.__class__.__name__ == 'RotatingFileHandler'
               for handler in logging.getLogger().handlers)
