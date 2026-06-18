#!/usr/bin/env bash
# deploy.sh — AmanAI guided deploy wrapper
# Usage: bash infra/deploy.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "=========================================="
echo "  AmanAI – AWS SAM Deploy"
echo "=========================================="
echo ""

# ── 1. Build ───────────────────────────────────
echo "[1/3] Building Lambda package..."
sam build --template-file template.yaml --use-container=false

# ── 2. Deploy (guided first time, uses samconfig.toml after) ──
echo ""
echo "[2/3] Deploying to AWS (us-east-1)..."
echo "      On first run you will be prompted for parameter values."
echo ""
sam deploy --guided --config-file samconfig.toml --template-file .aws-sam/build/template.yaml

# ── 3. Print post-deploy helper values ────────
echo ""
echo "[3/3] Fetching stack outputs..."
echo ""

STACK_NAME="amanai"
REGION="us-east-1"

get_output() {
  aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" \
    --output text 2>/dev/null || echo "(not found)"
}

API_URL=$(get_output "ApiUrl")
USER_POOL_ID=$(get_output "UserPoolId")
CLIENT_ID=$(get_output "UserPoolClientId")
HOSTED_UI=$(get_output "HostedUiDomain")
GUARDRAIL_ID=$(get_output "GuardrailId")
SECRET_ARN=$(get_output "SupabaseSecretArn")

echo "=========================================="
echo "  Paste these into Vercel Environment Variables"
echo "=========================================="
echo "NEXT_PUBLIC_API_BASE_URL=$API_URL"
echo "NEXT_PUBLIC_COGNITO_USER_POOL_ID=$USER_POOL_ID"
echo "NEXT_PUBLIC_COGNITO_CLIENT_ID=$CLIENT_ID"
echo "NEXT_PUBLIC_COGNITO_REGION=us-east-1"
echo "NEXT_PUBLIC_COGNITO_HOSTED_UI=$HOSTED_UI"
echo ""
echo "=========================================="
echo "  Step A: Store the Supabase service key in Secrets Manager"
echo "=========================================="
echo "Run (replace <YOUR_KEY> with your actual Supabase service role key):"
echo ""
echo "  aws secretsmanager put-secret-value \\"
echo "    --secret-id \"$SECRET_ARN\" \\"
echo "    --secret-string '{\"SUPABASE_SERVICE_KEY\":\"<YOUR_KEY>\"}' \\"
echo "    --region $REGION"
echo ""
echo "  Also set SUPABASE_URL in the Lambda environment:"
echo "  (Update the template.yaml SUPABASE_URL env var and redeploy, or use:"
echo "  aws lambda update-function-configuration \\"
echo "    --function-name amanai-backend-prod \\"
echo "    --environment 'Variables={SUPABASE_URL=https://<your-project>.supabase.co,...}')"
echo ""
echo "=========================================="
echo "  Step B: Create an admin user"
echo "=========================================="
echo "Run (replace <EMAIL> and <PASSWORD>):"
echo ""
echo "  aws cognito-idp admin-create-user \\"
echo "    --user-pool-id \"$USER_POOL_ID\" \\"
echo "    --username \"<EMAIL>\" \\"
echo "    --temporary-password \"TempPass1!\" \\"
echo "    --region $REGION"
echo ""
echo "  aws cognito-idp admin-add-user-to-group \\"
echo "    --user-pool-id \"$USER_POOL_ID\" \\"
echo "    --username \"<EMAIL>\" \\"
echo "    --group-name admin \\"
echo "    --region $REGION"
echo ""
echo "  Then sign in via the Hosted UI or frontend to set a permanent password."
echo ""
echo "=========================================="
echo "  Done!"
echo "=========================================="
