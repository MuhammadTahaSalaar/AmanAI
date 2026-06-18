import { Amplify } from 'aws-amplify'

const region       = process.env.NEXT_PUBLIC_COGNITO_REGION    ?? 'us-east-1'
const userPoolId   = process.env.NEXT_PUBLIC_COGNITO_USER_POOL_ID  ?? ''
const clientId     = process.env.NEXT_PUBLIC_COGNITO_CLIENT_ID ?? ''

Amplify.configure(
  {
    Auth: {
      Cognito: {
        userPoolId,
        userPoolClientId: clientId,
        signUpVerificationMethod: 'code',
      },
    },
  },
  { ssr: false },
)

export { region, userPoolId, clientId }
