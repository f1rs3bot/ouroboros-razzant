import assert from 'node:assert/strict';
import test from 'node:test';
import { taskArtifactDownloadUrl } from '../modules/api_client.js';

test('captured task files use the backend canonical URL encoding', () => {
    assert.equal(taskArtifactDownloadUrl('task', "résumé's (complete).zip"),
        '/api/tasks/task/artifacts/r%C3%A9sum%C3%A9%27s%20%28complete%29.zip');
    assert.equal(taskArtifactDownloadUrl('task', 'ordinary.bin'), '/api/tasks/task/artifacts/ordinary.bin');
    for (const name of ['../other', 'folder/file', 'folder\\file', '.artifact_manifest.json']) {
        assert.equal(taskArtifactDownloadUrl('task', name), '');
    }
    assert.equal(taskArtifactDownloadUrl('other/task', 'file.bin'), '');
});
