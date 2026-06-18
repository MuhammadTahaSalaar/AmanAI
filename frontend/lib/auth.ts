import {
  signIn,
  signUp,
  signOut,
  confirmSignUp,
  getCurrentUser,
  fetchAuthSession,
} from 'aws-amplify/auth'

export type AuthUser = {
  username: string
  email: string
  isAdmin: boolean
}

/** Returns the current Cognito ID token (JWT), or null if not signed in. */
export async function getIdToken(): Promise<string | null> {
  try {
    const session = await fetchAuthSession()
    return session.tokens?.idToken?.toString() ?? null
  } catch {
    return null
  }
}

/** Returns the current user with admin flag derived from cognito:groups claim. */
export async function getAuthUser(): Promise<AuthUser | null> {
  try {
    const user    = await getCurrentUser()
    const session = await fetchAuthSession()
    const payload = session.tokens?.idToken?.payload as Record<string, unknown> | undefined
    const groups  = (payload?.['cognito:groups'] as string[] | undefined) ?? []
    return {
      username: user.username,
      email:    (payload?.email as string | undefined) ?? user.username,
      isAdmin:  groups.includes('admin'),
    }
  } catch {
    return null
  }
}

export async function doSignIn(email: string, password: string) {
  return signIn({ username: email, password })
}

export async function doSignUp(email: string, password: string) {
  return signUp({
    username: email,
    password,
    options: { userAttributes: { email } },
  })
}

export async function doConfirmSignUp(email: string, code: string) {
  return confirmSignUp({ username: email, confirmationCode: code })
}

export async function doSignOut() {
  return signOut()
}
