# AmanAI Frontend

Next.js 14 (App Router, TypeScript) chat UI for the AmanAI banking RAG assistant.

## Run locally

```bash
cp .env.example .env.local   # fill in your values
npm install
npm run dev                  # http://localhost:3000
```

## Deploy on Vercel

1. Import the repo in Vercel — set **Root Directory** to `frontend`.
2. Add env vars from `.env.example` in the Vercel dashboard.
3. Deploy. Vercel auto-detects Next.js.

## Env vars

| Variable | Description |
|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | FastAPI backend URL (no trailing slash) |
| `NEXT_PUBLIC_COGNITO_REGION` | e.g. `us-east-1` |
| `NEXT_PUBLIC_COGNITO_USER_POOL_ID` | Cognito User Pool ID |
| `NEXT_PUBLIC_COGNITO_CLIENT_ID` | Cognito App Client ID |
| `NEXT_PUBLIC_COGNITO_DOMAIN` | Hosted UI domain (optional) |

## Production build verification

To verify the production build and TypeScript compilation, run on a clean environment or let Vercel handle it: `rm -rf node_modules package-lock.json && npm install && npm run build`. An in-sandbox dependency-hoisting quirk can cause `tsc` to report spurious missing exports from `aws-amplify/auth` that do not occur on a clean install; if you see those errors locally, the clean-install sequence above (or the Vercel build log) is the authoritative check.
