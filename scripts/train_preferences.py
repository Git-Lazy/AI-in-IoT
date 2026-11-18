"""Train the learned-preferences models and print what they learned.

    python scripts/train_preferences.py                    report on what's in the database
    python scripts/train_preferences.py --demo 6           first add 6 weeks of generated history
    python scripts/train_preferences.py --clear-demo       remove generated history

Uses the same database as the backend. If the backend is running, it retrains
on its own every time new arrivals come in; after --demo / --clear-demo press
"Retrain" on the dashboard (or restart it) so it picks the change up.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import init_db, session_scope  # noqa: E402
from app.db.models import Person  # noqa: E402
from app.learning import demo  # noqa: E402
from app.learning.model import MIN_SAMPLES, learner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--person", help="whose history to generate (default: the admin)")
    parser.add_argument("--demo", type=int, metavar="WEEKS", help="add this many weeks of generated history")
    parser.add_argument("--clear-demo", action="store_true", help="remove generated history")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")          # the ° in temperatures, on Windows consoles
    init_db()

    if args.clear_demo:
        print(f"removed {demo.clear()} generated arrival(s)")
    if args.demo:
        with session_scope() as s:
            q = s.query(Person)
            person = (q.filter(Person.name == args.person) if args.person
                      else q.filter(Person.role == "admin")).first()
            if person is None:
                sys.exit(f"no person {args.person or '(admin)'} -- check the name (it's case-sensitive)")
            pid, name = person.id, person.name
        print(f"added {demo.seed(pid, args.demo)} generated arrival(s) for {name}")

    models = learner.train()
    if not models:
        sys.exit(f"nothing to learn from yet: each person/device needs {MIN_SAMPLES}+ finished arrivals")

    for m in sorted(models, key=lambda m: (m.person, m.device)):
        verdict = "learned" if m.accuracy > m.baseline else "no better than the usual"
        print(f"\n{m.person} / {m.device}   ({m.samples} arrivals)")
        print(f"  accuracy {m.accuracy:.0%}  vs  always-the-usual {m.baseline:.0%}   -> {verdict}")
        for r in m.rules():
            print(f"    {', '.join(r['when']):<52} -> {r['wording']:<16} "
                  f"({r['arrivals']} arrivals, {r['share']:.0%})")


if __name__ == "__main__":
    main()
