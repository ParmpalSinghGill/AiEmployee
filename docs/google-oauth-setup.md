# Google setup — let anyone connect their Gmail & Calendar

Users of your AI Employee connect their own Google accounts with one click and
a Google consent page — no keys on their side. You, the operator, register the
app with Google **once**:

## 1. Create the OAuth client (once, ~10 min, free)

1. Go to [console.cloud.google.com](https://console.cloud.google.com) → create
   (or pick) a project.
2. **APIs & Services → Library**: enable **Gmail API** and **Google Calendar
   API**.
3. **APIs & Services → OAuth consent screen**: choose **External**, fill in the
   app name (what users see on the consent page) and your contact email.
4. **Credentials → Create credentials → OAuth client ID** → application type
   **Desktop app** (works for the localhost console; use **Web application**
   with redirect URI `https://your-host/ui/api/google/oauth/callback` when you
   deploy behind a domain).
5. Download the JSON and save it as **`data/google_oauth_client.json`** in this
   repo (chmod 600 recommended). Restart the server.

## 2. Who can connect

| Publishing status | Who can click "Connect Google account" |
|---|---|
| **Testing** (default) | Up to 100 **test users** you list under *OAuth consent screen → Test users*. Add each person's Gmail address there. |
| **In production, unverified** | Anyone, but Google shows an "unverified app" warning and caps restricted (Gmail-read) grants. |
| **Verified** | Anyone, clean consent screen. Requires Google's app review; Gmail *read* scopes additionally need the annual CASA security assessment. `gmail.send` + Calendar are only "sensitive" — review, no paid assessment. |

Tip: to defer the CASA cost, remove `gmail.readonly` from
`GOOGLE_OAUTH_SCOPES` in `.env` and ship send-only + Calendar first.

## 3. Using it

- Console → **🔌 MCP → Google accounts → Connect Google account** → approve on
  Google's page. Done: the `gmail_search`, `gmail_read`, `gmail_send`,
  `calendar_list_events`, `calendar_create_event` tools go live.
- Several accounts can be connected; tools take an optional `account` email.
- Sending mail and creating events are **approval-gated** — the run pauses for
  a human ✓ before anything leaves the building.
- Tokens live in `data/google_accounts.json` (owner-only file mode) and refresh
  automatically; **Disconnect** revokes the grant at Google.
