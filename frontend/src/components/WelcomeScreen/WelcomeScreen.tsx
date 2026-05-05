import { useState, useCallback, type ReactNode } from 'react';
import {
  Box,
  Typography,
  Button,
  CircularProgress,
  Alert,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import LoginIcon from '@mui/icons-material/Login';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import { useSessionStore } from '@/store/sessionStore';
import { useAgentStore } from '@/store/agentStore';
import { apiFetch } from '@/utils/api';
import { triggerLogin } from '@/hooks/useAuth';

const DBX_ORANGE = '#FF3621';

// ---------------------------------------------------------------------------
// ChecklistStep sub-component
// ---------------------------------------------------------------------------

type StepStatus = 'completed' | 'active' | 'locked';

interface ChecklistStepProps {
  stepNumber: number;
  title: string;
  description: string;
  status: StepStatus;
  lockedReason?: string;
  actionLabel?: string;
  onAction?: () => void;
  actionIcon?: ReactNode;
  loading?: boolean;
  isLast?: boolean;
}

function StepIndicator({ status, stepNumber }: { status: StepStatus; stepNumber: number }) {
  if (status === 'completed') {
    return <CheckCircleIcon sx={{ fontSize: 28, color: 'var(--accent-green)' }} />;
  }
  return (
    <Box
      sx={{
        width: 28,
        height: 28,
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '0.8rem',
        fontWeight: 700,
        ...(status === 'active'
          ? { bgcolor: DBX_ORANGE, color: '#fff' }
          : { bgcolor: 'transparent', border: '2px solid var(--border)', color: 'var(--muted-text)' }),
      }}
    >
      {stepNumber}
    </Box>
  );
}

function ChecklistStep({
  stepNumber,
  title,
  description,
  status,
  lockedReason,
  actionLabel,
  onAction,
  actionIcon,
  loading = false,
  isLast = false,
}: ChecklistStepProps) {
  const btnSx = {
    px: 3,
    py: 0.75,
    fontSize: '0.85rem',
    fontWeight: 700,
    textTransform: 'none' as const,
    borderRadius: '10px',
    whiteSpace: 'nowrap' as const,
    textDecoration: 'none',
    ...(status === 'active'
      ? {
          bgcolor: DBX_ORANGE,
          color: '#fff',
          boxShadow: '0 2px 12px rgba(255, 54, 33, 0.25)',
          '&:hover': { bgcolor: '#FF5A47', boxShadow: '0 4px 20px rgba(255, 54, 33, 0.4)' },
        }
      : {
          bgcolor: 'rgba(255,255,255,0.04)',
          color: 'var(--muted-text)',
          '&.Mui-disabled': { bgcolor: 'rgba(255,255,255,0.04)', color: 'var(--muted-text)' },
        }),
  };

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 2,
        px: 3,
        py: 2.5,
        borderLeft: '3px solid',
        borderLeftColor:
          status === 'completed'
            ? 'var(--accent-green)'
            : status === 'active'
              ? DBX_ORANGE
              : 'transparent',
        ...(!isLast && { borderBottom: '1px solid var(--border)' }),
        opacity: status === 'locked' ? 0.55 : 1,
        transition: 'opacity 0.2s, border-color 0.2s',
      }}
    >
      <StepIndicator status={status} stepNumber={stepNumber} />

      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Typography
          variant="subtitle2"
          sx={{
            fontWeight: 600,
            fontSize: '0.92rem',
            color: status === 'completed' ? 'var(--muted-text)' : 'var(--text)',
            ...(status === 'completed' && { textDecoration: 'line-through', textDecorationColor: 'var(--muted-text)' }),
          }}
        >
          {title}
        </Typography>
        <Typography variant="body2" sx={{ color: 'var(--muted-text)', fontSize: '0.8rem', mt: 0.25, lineHeight: 1.5 }}>
          {status === 'locked' && lockedReason ? lockedReason : description}
        </Typography>
      </Box>

      {status === 'completed' ? (
        <Typography variant="caption" sx={{ color: 'var(--accent-green)', fontWeight: 600, fontSize: '0.78rem', whiteSpace: 'nowrap' }}>
          Done
        </Typography>
      ) : actionLabel ? (
        <Button
          variant="contained"
          size="small"
          disabled={status === 'locked' || loading}
          startIcon={loading ? <CircularProgress size={16} color="inherit" /> : actionIcon}
          onClick={onAction}
          sx={btnSx}
        >
          {loading ? 'Loading...' : actionLabel}
        </Button>
      ) : null}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// WelcomeScreen — Databricks Apps deploy. The Apps proxy authenticates the
// user at the edge and forwards X-Forwarded-Access-Token, so by the time the
// FE renders, identity is already resolved. Local dev falls back to the SDK
// auth chain via /auth/me.
// ---------------------------------------------------------------------------

export default function WelcomeScreen() {
  const { createSession } = useSessionStore();
  const { setPlan, clearPanel, user } = useAgentStore();
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isAuthenticated = !!user?.authenticated;

  const handleStartSession = useCallback(async () => {
    if (isCreating) return;
    setIsCreating(true);
    setError(null);

    try {
      const response = await apiFetch('/api/session', { method: 'POST' });
      if (response.status === 503) {
        const data = await response.json();
        setError(data.detail || 'Server is at capacity. Please try again later.');
        return;
      }
      if (response.status === 401) {
        triggerLogin();
        return;
      }
      if (!response.ok) {
        setError('Failed to create session. Please try again.');
        return;
      }
      const data = await response.json();
      createSession(data.session_id);
      setPlan([]);
      clearPanel();
    } catch {
      // Redirect may throw — ignore
    } finally {
      setIsCreating(false);
    }
  }, [isCreating, createSession, setPlan, clearPanel]);

  const signInStatus: StepStatus = isAuthenticated ? 'completed' : 'active';
  const startStatus: StepStatus = isAuthenticated ? 'active' : 'locked';

  return (
    <Box
      sx={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--body-gradient)',
        py: 8,
      }}
    >
      <Box
        component="img"
        src="/smolagents.webp"
        alt="ml-intern"
        sx={{ width: 80, height: 80, mb: 2.5, display: 'block' }}
      />

      <Typography
        variant="h2"
        sx={{
          fontWeight: 800,
          color: 'var(--text)',
          mb: 1,
          letterSpacing: '-0.02em',
          fontSize: { xs: '1.8rem', md: '2.4rem' },
        }}
      >
        ML Intern
      </Typography>

      <Typography
        variant="body1"
        sx={{
          color: 'var(--muted-text)',
          maxWidth: 480,
          mb: 4,
          lineHeight: 1.7,
          fontSize: '0.9rem',
          textAlign: 'center',
          px: 2,
          '& strong': { color: 'var(--text)', fontWeight: 600 },
        }}
      >
        Your personal <strong>ML agent</strong> on Databricks. Reads <strong>papers</strong>,
        ingests <strong>UC datasets</strong>, runs <strong>jobs</strong>, registers <strong>UC models</strong>,
        and iterates until the numbers go up.
      </Typography>

      <Box
        sx={{
          width: '100%',
          maxWidth: 520,
          bgcolor: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: '12px',
          overflow: 'hidden',
          mx: 2,
        }}
      >
        {isAuthenticated ? (
          <ChecklistStep
            stepNumber={1}
            title="Start Session"
            description="Launch an agent session connected to your Databricks workspace."
            status="active"
            actionLabel="Start Session"
            actionIcon={<RocketLaunchIcon sx={{ fontSize: 16 }} />}
            onAction={handleStartSession}
            loading={isCreating}
            isLast
          />
        ) : (
          <>
            <ChecklistStep
              stepNumber={1}
              title="Sign in"
              description="Authenticate to your Databricks workspace."
              status={signInStatus}
              actionLabel="Sign in"
              actionIcon={<LoginIcon sx={{ fontSize: 16 }} />}
              onAction={() => triggerLogin()}
            />
            <ChecklistStep
              stepNumber={2}
              title="Start Session"
              description="Launch an agent session connected to your Databricks workspace."
              status={startStatus}
              lockedReason="Sign in first to continue."
              actionLabel="Start Session"
              actionIcon={<RocketLaunchIcon sx={{ fontSize: 16 }} />}
              onAction={handleStartSession}
              loading={isCreating}
              isLast
            />
          </>
        )}
      </Box>

      {error && (
        <Alert
          severity="warning"
          variant="outlined"
          onClose={() => setError(null)}
          sx={{
            mt: 3,
            maxWidth: 400,
            fontSize: '0.8rem',
            borderColor: DBX_ORANGE,
            color: 'var(--text)',
          }}
        >
          {error}
        </Alert>
      )}

      <Typography
        variant="caption"
        sx={{ mt: 4, color: 'var(--muted-text)', opacity: 0.5, fontSize: '0.7rem' }}
      >
        Sessions persist in Lakebase. Telemetry via MLflow Tracing.
      </Typography>
    </Box>
  );
}
