import { type MutableRefObject, useCallback } from 'react'

import { PROMPT_SUBMIT_REQUEST_TIMEOUT_MS } from '@/hermes'
import type { Translations } from '@/i18n'
import { type ChatMessage, textPart } from '@/lib/chat-messages'
import { optimisticAttachmentRef } from '@/lib/chat-runtime'
import { setMutableRef } from '@/lib/mutable-ref'
import {
  $composerAttachments,
  clearComposerAttachments,
  type ComposerAttachment,
  terminalContextBlocksFromDraft
} from '@/store/composer'
import { clearNotifications, notify, notifyError } from '@/store/notifications'
import { requestDesktopOnboarding } from '@/store/onboarding'
import { setAwaitingResponse, setBusy, setMessages } from '@/store/session'

import type { ClientSessionState } from '../../../types'

import {
  _submitInFlight,
  type GatewayRequest,
  inlineErrorMessage,
  isGatewayTimeoutError,
  isProviderSetupError,
  isSessionBusyError,
  isSessionNotFoundError,
  type SubmitTextOptions,
  withSessionBusyRetry
} from './utils'

function logFreshSubmitDiagnostic(phase: string, details: Record<string, unknown> = {}) {
  // Electron forwards renderer console.error entries to desktop.log. Keep this
  // deliberately scoped to fresh-session submits and never include prompt text
  // so it is useful for diagnosing the first-message race without logging user
  // content. The bridge check keeps unit-test output quiet and limits this to
  // the Desktop surface where the bug occurs.
  if (typeof window === 'undefined' || !window.hermesDesktop) {
    return
  }

  try {
    console.error(`[hermes-submit-diagnostic] ${phase} ${JSON.stringify(details)}`)
  } catch {
    // Diagnostics must never affect the submit path.
  }
}

interface SubmitPromptDeps {
  activeSessionId: string | null
  activeSessionIdRef: MutableRefObject<string | null>
  busyRef: MutableRefObject<boolean>
  copy: Translations['desktop']
  createBackendSessionForSend: (preview?: string | null) => Promise<string | null>
  getRouteToken: () => string
  requestGateway: GatewayRequest
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  syncAttachmentsForSubmit: (
    sessionId: string,
    attachments: ComposerAttachment[],
    options?: { updateComposerAttachments?: boolean }
  ) => Promise<ComposerAttachment[]>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}

/** The prompt submit pipeline, extracted from usePromptActions. */
export function useSubmitPrompt(deps: SubmitPromptDeps) {
  const {
    activeSessionId,
    activeSessionIdRef,
    busyRef,
    copy,
    createBackendSessionForSend,
    getRouteToken,
    requestGateway,
    selectedStoredSessionIdRef,
    syncAttachmentsForSubmit,
    updateSessionState
  } = deps

  return useCallback(
    async (rawText: string, options?: SubmitTextOptions) => {
      const visibleText = rawText.trim()
      const usingComposerAttachments = !options?.attachments

      // Drop undefined/null holes a session switch or draft restore can leave in
      // the attachments array (same bug class as AttachmentList #49624). Without
      // this, the sibling iterations below (a.kind / a.label / a.refText, and the
      // sync step) throw "Cannot read properties of undefined (reading 'refText')"
      // and break the chat surface.
      const attachments = (options?.attachments ?? $composerAttachments.get()).filter((a): a is ComposerAttachment =>
        Boolean(a)
      )

      const terminalContextBlocks = terminalContextBlocksFromDraft(rawText).join('\n\n')
      const hasImage = attachments.some(a => a.kind === 'image')

      // Refs are recomputed after sync (file.attach rewrites @file: refs to
      // workspace-relative paths the remote gateway can resolve). Seed the
      // optimistic message with the pre-sync refs, then rewrite once synced.
      // Images use their base64 preview so the thumbnail renders inline without
      // a (remote-mode 403-prone) /api/media fetch — see optimisticAttachmentRef.
      let attachmentRefs = attachments.map(optimisticAttachmentRef).filter((r): r is string => Boolean(r))

      const buildContextText = (atts: ComposerAttachment[]): string => {
        // atts may be the post-sync array, which can reintroduce holes; filter
        // before touching a.refText / a.kind.
        const present = atts.filter((a): a is ComposerAttachment => Boolean(a))

        const contextRefs = present
          .map(a => a.refText)
          .filter(Boolean)
          .join('\n')

        return (
          [contextRefs, terminalContextBlocks, visibleText].filter(Boolean).join('\n\n') ||
          (present.some(a => a.kind === 'image') ? 'What do you see in this image?' : '')
        )
      }

      // Queue drains fire on the busy→false settle edge, where busyRef (synced
      // from $busy by a separate effect) may still read true — honoring it would
      // bounce the drained send. The drain lock serializes them; the user path
      // keeps the guard so a stray Enter mid-turn can't double-submit.
      const hasSendable = Boolean(visibleText || terminalContextBlocks || attachments.length || hasImage)

      if (!hasSendable || (!options?.fromQueue && busyRef.current)) {
        return false
      }

      // Pin the session context for the whole async submit pipeline. Without
      // this, a fast session switch during session.resume / file.attach can
      // redirect the user's text into a different chat (#54527). Mutable —
      // not const — because a new-chat submit legitimately re-homes to the
      // session it creates (see the re-pin after createBackendSessionForSend).
      const startingActiveSessionId = activeSessionIdRef.current
      const startingStoredSessionId = selectedStoredSessionIdRef.current
      const startingRouteToken = getRouteToken()
      const isFreshSessionSubmit = !startingActiveSessionId && !startingStoredSessionId

      const diagnostic = (phase: string, details: Record<string, unknown> = {}) => {
        if (isFreshSessionSubmit) {
          logFreshSubmitDiagnostic(phase, details)
        }
      }

      diagnostic('submit-start', {
        activeSessionId: startingActiveSessionId,
        routeToken: startingRouteToken,
        storedSessionId: startingStoredSessionId
      })

      // Creating/resuming a runtime session is part of this submit pipeline.
      // Those operations intentionally update all three values below (the
      // runtime id, the stored id, and, for a new chat, the route). Keep the
      // context guard pinned to the *latest accepted binding* instead of the
      // pre-create draft values, otherwise the expected new-session transition
      // looks like a user switch and the first prompt is silently aborted.
      let expectedActiveSessionId = startingActiveSessionId
      let expectedStoredSessionId = startingStoredSessionId
      let expectedRouteToken = startingRouteToken

      const sessionContextDrifted = (): boolean =>
        activeSessionIdRef.current !== expectedActiveSessionId ||
        selectedStoredSessionIdRef.current !== expectedStoredSessionId ||
        getRouteToken() !== expectedRouteToken

      const adoptSessionBinding = (runtimeSessionId: string) => {
        expectedActiveSessionId = runtimeSessionId
        expectedStoredSessionId = selectedStoredSessionIdRef.current
        expectedRouteToken = getRouteToken()
      }

      // One submit in flight per session — drop any concurrent re-fire so a
      // stalled turn can't stack the same prompt into multiple real turns.
      const submitLockKey = startingStoredSessionId || startingActiveSessionId || '__pending_new__'

      if (_submitInFlight.has(submitLockKey)) {
        return false
      }

      _submitInFlight.add(submitLockKey)
      let submitLockReleased = false

      const releaseSubmitLock = () => {
        if (!submitLockReleased) {
          submitLockReleased = true
          _submitInFlight.delete(submitLockKey)
        }
      }

      const optimisticId = `user-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

      const buildUserMessage = (): ChatMessage => ({
        id: optimisticId,
        role: 'user',
        parts: [textPart(visibleText || (attachmentRefs.length ? '' : attachments.map(a => a.label).join(', ')))],
        attachmentRefs
      })

      const releaseBusy = () => {
        releaseSubmitLock()
        setMutableRef(busyRef, false)
        setBusy(false)
        setAwaitingResponse(false)
      }

      // Idempotent optimistic insert — re-running with the resolved sessionId
      // after createBackendSessionForSend just overwrites with the same id.
      const seedOptimistic = (sid: string) =>
        updateSessionState(
          sid,
          state => ({
            ...state,
            messages: state.messages.some(m => m.id === optimisticId)
              ? state.messages
              : [...state.messages, buildUserMessage()],
            busy: true,
            awaitingResponse: true,
            pendingBranchGroup: null,
            sawAssistantPayload: false,
            // Fresh submit = new turn — clear any leftover interrupt flag, else
            // mutateStream/completeAssistantMessage drop every delta of this turn
            // (what made drained-after-interrupt sends go silent).
            interrupted: false
          }),
          expectedStoredSessionId
        )

      // After sync rewrites refs, refresh the optimistic message in place so the
      // transcript shows the resolved @file: ref rather than the local path.
      const rewriteOptimistic = (sid: string) =>
        updateSessionState(
          sid,
          state => ({
            ...state,
            messages: state.messages.map(message => (message.id === optimisticId ? buildUserMessage() : message))
          }),
          expectedStoredSessionId
        )

      const dropOptimistic = (sid: null | string) => {
        if (!sid) {
          setMessages(current => current.filter(m => m.id !== optimisticId))

          return
        }

        updateSessionState(
          sid,
          state => ({
            ...state,
            messages: state.messages.filter(m => m.id !== optimisticId),
            busy: false,
            awaitingResponse: false,
            pendingBranchGroup: null
          }),
          expectedStoredSessionId
        )
      }

      const abortForSessionSwitch = (optimisticSessionId: null | string): false => {
        diagnostic('abort-session-context', {
          currentActiveSessionId: activeSessionIdRef.current,
          currentRouteToken: getRouteToken(),
          currentStoredSessionId: selectedStoredSessionIdRef.current,
          expectedActiveSessionId,
          expectedRouteToken,
          expectedStoredSessionId,
          optimisticSessionId
        })
        dropOptimistic(optimisticSessionId)
        releaseBusy()

        return false
      }

      setMutableRef(busyRef, true)
      setBusy(true)
      setAwaitingResponse(true)
      clearNotifications()

      let sessionId: null | string = activeSessionId

      if (sessionId) {
        seedOptimistic(sessionId)
      } else {
        setMessages(current => [...current, buildUserMessage()])
      }

      if (!sessionId && startingStoredSessionId) {
        // A stored session is SELECTED but its runtime binding is gone (the
        // live session was orphan-reaped, or a timeout/reconnect cleared
        // activeSessionId). Continuing the selected conversation must mean
        // resuming it — minting a brand-new backend session here silently
        // splits the user's chat in two (#55578 symptom b). Only fall through
        // to session creation when NO stored session is selected (a genuine
        // new-chat draft).
        try {
          const resumed = await requestGateway<{ session_id: string }>('session.resume', {
            session_id: startingStoredSessionId
          })

          if (sessionContextDrifted()) {
            return abortForSessionSwitch(sessionId)
          }

          if (resumed?.session_id) {
            sessionId = resumed.session_id
            activeSessionIdRef.current = sessionId
            adoptSessionBinding(sessionId)
          }
        } catch {
          // Resume failed (session gone from state.db, gateway hiccup) —
          // fall through to creating a fresh session rather than dead-ending
          // the user's message.
        }

        if (sessionContextDrifted()) {
          return abortForSessionSwitch(sessionId)
        }

        if (sessionId) {
          seedOptimistic(sessionId)
        }
      }

      if (!sessionId) {
        diagnostic('session-create-start')

        try {
          sessionId = await createBackendSessionForSend(visibleText)
        } catch (err) {
          diagnostic('session-create-error', {
            errorType: err instanceof Error ? err.name : typeof err
          })
          dropOptimistic(null)
          releaseBusy()
          notifyError(err, copy.sessionUnavailable)

          return false
        }

        if (!sessionId) {
          // createBackendSessionForSend returns null when the user switched
          // sessions mid-create (it closes the orphaned session itself) —
          // abort silently. Anything else is a real failure worth a toast.
          if (sessionContextDrifted()) {
            return abortForSessionSwitch(null)
          }

          dropOptimistic(null)
          releaseBusy()
          notify({ kind: 'error', title: copy.sessionUnavailable, message: copy.createSessionFailed })

          return false
        }

        diagnostic('session-create-result', {
          currentActiveSessionId: activeSessionIdRef.current,
          currentRouteToken: getRouteToken(),
          currentStoredSessionId: selectedStoredSessionIdRef.current,
          runtimeSessionId: sessionId
        })

        // A successful create deliberately changes the selected stored id and
        // navigates from /new to /:stored-id. Rebase the guard before the
        // attachment sync and prompt RPC so that transition is not treated as
        // an external session switch.
        if (activeSessionIdRef.current && activeSessionIdRef.current !== sessionId) {
          return abortForSessionSwitch(sessionId)
        }

        // The real create action binds this ref before returning. Keeping the
        // assignment here also makes the submit contract explicit and avoids a
        // stale render/ref pair from dropping the prompt between create and
        // submit.
        activeSessionIdRef.current = sessionId
        adoptSessionBinding(sessionId)
        seedOptimistic(sessionId)
      }

      try {
        const syncedAttachments = await syncAttachmentsForSubmit(sessionId, attachments, {
          updateComposerAttachments: usingComposerAttachments
        })

        if (sessionContextDrifted()) {
          return abortForSessionSwitch(sessionId)
        }

        // Rewrite the optimistic message + prompt text with the synced refs so
        // the gateway receives @file: paths that resolve in its workspace.
        // (Images keep their inline base64 preview — see optimisticAttachmentRef.)
        attachmentRefs = syncedAttachments.map(optimisticAttachmentRef).filter((r): r is string => Boolean(r))
        rewriteOptimistic(sessionId)
        const text = buildContextText(syncedAttachments)

        diagnostic('prompt-submit-start', {
          currentActiveSessionId: activeSessionIdRef.current,
          currentRouteToken: getRouteToken(),
          currentStoredSessionId: selectedStoredSessionIdRef.current,
          runtimeSessionId: sessionId
        })

        // On sleep/wake the gateway's in-memory session may have been cleared
        // while the desktop app still holds the old session ID. Detect this,
        // resume the stored session to re-register it, and retry once.
        let submitErr: unknown = null

        try {
          await withSessionBusyRetry(() =>
            requestGateway('prompt.submit', { session_id: sessionId, text }, PROMPT_SUBMIT_REQUEST_TIMEOUT_MS)
          )
        } catch (firstErr) {
          if (
            (isSessionNotFoundError(firstErr) || isGatewayTimeoutError(firstErr)) &&
            startingStoredSessionId
          ) {
            // Re-register the session in the gateway and get a fresh live ID.
            // Timeouts recover the same way as "session not found": a starved
            // backend loop (#55578 symptom d) rejects the submit even though
            // the stored session is fine — resume + retry instead of erroring
            // out and losing the session binding.
            const resumed = await requestGateway<{ session_id: string }>('session.resume', {
              session_id: startingStoredSessionId,
              source: 'desktop'
            })

            if (sessionContextDrifted()) {
              return abortForSessionSwitch(sessionId)
            }

            const recoveredId = resumed?.session_id

            if (recoveredId) {
              activeSessionIdRef.current = recoveredId
              adoptSessionBinding(recoveredId)
              await withSessionBusyRetry(() =>
                requestGateway('prompt.submit', { session_id: recoveredId, text }, PROMPT_SUBMIT_REQUEST_TIMEOUT_MS)
              )
            } else {
              submitErr = firstErr
            }
          } else {
            submitErr = firstErr
          }
        }

        if (submitErr !== null) {
          throw submitErr
        }

        if (usingComposerAttachments) {
          clearComposerAttachments()
        }

        // Submit landed — the turn now runs (busy stays true), but the submit
        // window is closed, so release the lock for the next (sequential) send.
        releaseSubmitLock()
        diagnostic('prompt-submit-accepted', { runtimeSessionId: sessionId })

        return true
      } catch (err) {
        diagnostic('prompt-submit-error', {
          errorType: err instanceof Error ? err.name : typeof err,
          runtimeSessionId: sessionId
        })
        releaseBusy()

        // A queued drain that raced a not-yet-settled turn gets a transient
        // "session busy" (4009). Don't surface an error bubble/toast — the entry
        // stays queued and the composer's bounded auto-drain retries when idle.
        if (options?.fromQueue && isSessionBusyError(err)) {
          return false
        }

        const message = inlineErrorMessage(err, copy.promptFailed)

        updateSessionState(sessionId, state => ({
          ...state,
          messages: [
            ...state.messages,
            {
              id: `assistant-error-${Date.now()}`,
              role: 'assistant',
              parts: [],
              error: message || copy.promptFailed,
              branchGroupId: state.pendingBranchGroup ?? undefined
            }
          ],
          busy: false,
          awaitingResponse: false,
          pendingBranchGroup: null,
          sawAssistantPayload: true
        }))

        if (isProviderSetupError(err)) {
          requestDesktopOnboarding(copy.providerCredentialRequired)

          return false
        }

        notifyError(err, copy.promptFailed)

        return false
      }
    },
    [
      activeSessionId,
      activeSessionIdRef,
      busyRef,
      copy,
      createBackendSessionForSend,
      getRouteToken,
      requestGateway,
      selectedStoredSessionIdRef,
      syncAttachmentsForSubmit,
      updateSessionState
    ]
  )
}
