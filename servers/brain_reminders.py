"""brain — change-impact lookup ('if you modify this file, also check ...')."""

from typing import Any, Dict, List
import json


class BrainRemindersMixin:
    """Change-impact lookup mixin for Brain."""

    def get_change_impact(self, file_path: str) -> List[Dict[str, Any]]:
        """Return all change impact entries for a file — 'If you modify this, also check...'"""
        results = []
        # Search impact nodes
        cur = self.conn.execute(
            "SELECT id, title, content FROM nodes WHERE type = 'impact' AND archived = 0 AND content LIKE ?",
            (f'%{file_path}%',)
        )
        for row in cur.fetchall():
            results.append({'id': row[0], 'title': row[1], 'content': row[2]})

        # Also check change_impacts in metadata KV
        cur = self.conn.execute(
            "SELECT kv.node_id, n.title, kv.value FROM node_metadata_kv kv "
            "JOIN nodes n ON n.id = kv.node_id "
            "WHERE kv.key = 'change_impacts' AND kv.value LIKE ?",
            (f'%{file_path}%',)
        )
        for row in cur.fetchall():
            try:
                impacts = json.loads(row[2])
                for imp in impacts:
                    if file_path in imp.get('if_modified', '') or file_path in imp.get('must_check', ''):
                        results.append({'id': row[0], 'title': row[1], 'impact': imp})
            except (json.JSONDecodeError, TypeError):
                pass
        return results
