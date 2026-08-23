# Deployment runbook

How to put Divot into production and what to check before each release. Items
already handled in code are marked so, with a pointer to where.

## 1. Configuration and secrets

The app has no API keys or third-party credentials. Everything configurable is
an environment variable read in `config.py` and `utils/observability.py`.

1. Copy `.env.example` to `.env` on the host and set the values. `.env` is
   git-ignored; never commit it.
2. Point `POTHOLE_DATABASE_PATH` at a persistent volume. The default lives
   inside the app directory and is lost on redeploy.
3. Point `POTHOLE_MODEL_PATH` at the weights file. It is ~19 MB and ships in
   `models/best.pt`; mount it separately if you rotate models without redeploying.
4. Set `STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=none` in production. Without it
   Streamlit renders the traceback in the browser. Errors still reach the log
   through `report_error`, which shows the user a short reference id instead.

## 2. TLS

Streamlit does not terminate TLS. Run it behind a reverse proxy (nginx, Caddy,
or the platform's ingress) that does, and redirect port 80 to 443.

Passwords cross the wire on sign-in. Over plain HTTP they are readable by
anyone on the path, and the session state that keeps a user signed in is
equally exposed. **Do not expose this app on HTTP.**

The proxy must forward the original `Host` header and the WebSocket upgrade
headers, or Streamlit's XSRF check will reject the connection:

```
proxy_set_header Host $host;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
```

## 3. Access control — already in code

- Passwords are stored as per-user salted scrypt hashes; see `utils/auth.py`.
- Sign-in failures return one message for both unknown user and wrong password,
  and spend comparable time, so the form cannot enumerate accounts.
- Five failed attempts lock an account for 15 minutes (`MAX_FAILED_ATTEMPTS`,
  `LOCKOUT_MINUTES`).
- Every history read and write is scoped to one `user_id` (`utils/storage.py`),
  and each page re-asserts the session with `require_user()` (`utils/session.py`).

What is **not** in code: there are no roles. Every account can do everything to
its own data and nothing to anyone else's. If you need an operator role, it does
not exist yet — do not assume one.

## 4. Pre-release checks

Run locally or in CI (`.github/workflows/ci.yml` runs all three on every push
and pull request):

```bash
ruff check . && black --check . && pytest -q
```

Then, on a staging instance with production settings:

- Sign up, sign out, sign back in.
- Run one image detection and one video detection end to end.
- Confirm the history page shows only that account's records.
- Force an error (point `POTHOLE_MODEL_PATH` at a missing file) and confirm the
  browser shows a reference id, **not** a traceback.
- Check the layout on a narrow viewport; the upload and result panes are the
  paths that matter.

## 5. Performance

- The detections table is indexed on `(user_id, created_at DESC)`, which is the
  only query shape the history page issues (`utils/storage.py`).
- Uploads are capped before any decoding: 10 MB images, 200 MB / 5 minute video
  (`config.py`, enforced in `utils/validators.py`). `maxUploadSize` in
  `.streamlit/config.toml` matches.
- Model weights load once per process and are reused; a cold start pays for the
  load, subsequent requests do not.
- Video processing is CPU-bound and slow. One long upload occupies a worker for
  its whole duration. Size the instance for concurrent inspections, or expect
  queueing.

Rate limiting: the login lockout covers credential guessing, but nothing limits
upload volume. Put a request limit at the proxy — a modest cap on the upload
endpoint is enough to keep one client from saturating the box.

## 6. Logging and monitoring

`configure_logging()` runs at startup and writes to stderr, so whatever
supervises the process collects the logs. Set `POTHOLE_LOG_LEVEL` to adjust.

Logged today: sign-in, sign-out, registration, refused attempts with the
reason (never the password), and every caught detection failure with a
reference id and traceback.

Not built in: uptime alerting and error aggregation. Point the platform's
monitor at `/` (Streamlit answers 200 when healthy) and ship stderr to whatever
log service you use.

## 7. Backups

```bash
python scripts/backup_database.py --output-dir /var/backups/divot --keep 30
```

Uses SQLite's online backup API, so it is safe while the app is running, then
runs an integrity check and prunes old copies. Schedule it daily.

Restore is a file copy: stop the app, put the backup at
`POTHOLE_DATABASE_PATH`, start it. **Test this on staging before you need it.**
The database holds accounts and history; generated media under `outputs/` is
reproducible and not backed up.

## 8. Rollback

The app is stateless apart from the database. To roll back, deploy the previous
commit — schema changes so far have been additive (`initialize_database` adds
`user_id` if missing), so an older build reads a newer database without failing.
If a release does change the schema destructively, take a backup first and say
so in the release notes.

## 9. Streamlit Community Cloud

The fastest way to get a public URL, with one limitation that decides whether
it is suitable at all.

### The storage caveat — read this first

**Community Cloud has no persistent disk.** The container is rebuilt on every
redeploy, on every reboot, and after the app sleeps from inactivity. When that
happens `database/potholes.db` is gone, which means **every account and every
saved detection is erased**. People will find themselves unable to sign in with
the password they just set.

That is fine for a demo or a portfolio link. It is not fine for anything real.
For durable accounts you need a host with a mounted volume — a small VM, Fly.io
with a volume, Railway, or a container platform — following sections 1 to 8.
`scripts/backup_database.py` cannot save you here either: it writes to the same
disappearing disk.

If you deploy to Community Cloud anyway, say so on the sign-in screen so nobody
treats the account as permanent.

### Steps

1. Push this branch and open <https://share.streamlit.io>, then sign in with
   the GitHub account that owns the repository.
2. **New app** -> **Deploy a public app from GitHub**, and select:
   - Repository: `ac708726-coder/POT-HOLE-DETECTION7`
   - Branch: `deploy/streamlit-cloud`
   - Main file path: `app.py`
   - Python version: 3.12
3. Under **Advanced settings**, paste into the secrets box:

   ```toml
   POTHOLE_LOG_LEVEL = "INFO"
   ```

   Streamlit exposes secrets as environment variables, which is what
   `config.py` reads. Leave the paths at their defaults; the model ships in the
   repository and the database is created on first run.
4. Deploy. The first build takes several minutes — it installs torch.

### What this branch changes for the platform

`requirements.txt` pins the CPU torch index and the headless OpenCV build.
Without both, the build either exceeds the resource limit or fails at import on
a missing `libGL`. If the build still reports a missing shared library, add a
`packages.txt` at the repository root naming the apt package.

### Expectations on the free tier

- Roughly 1 GB of memory. The model loads in well under that, but video
  processing on a long clip can approach it — keep test uploads short.
- No GPU. Image detection takes a few seconds; a minute of video takes minutes.
- The app sleeps after inactivity and cold-starts on the next visit, reloading
  the model. First request after a sleep is slow — and, per the caveat above,
  starts from an empty database.
- HTTPS is provided by the platform, so section 2 is already satisfied.
  Tracebacks are still worth hiding: set `STREAMLIT_CLIENT_SHOW_ERROR_DETAILS`
  to `none` in the secrets box before sharing the link widely.
