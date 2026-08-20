from app.core.storage import LocalStorageBackend


async def test_local_storage_round_trip(tmp_path) -> None:
    backend = LocalStorageBackend(str(tmp_path))

    path = await backend.save("docs/a.txt", b"hello atlas")
    assert await backend.read("docs/a.txt") == b"hello atlas"
    assert path.endswith("a.txt")

    await backend.delete("docs/a.txt")
    assert not (tmp_path / "docs" / "a.txt").exists()
