import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { LandingPage } from './components/LandingPage';
import { Terminal } from './components/Terminal';
import { SignalHistory } from './components/SignalHistory';
import { Watchlist } from './components/Watchlist';
import { AdminLogs } from './components/AdminLogs';
import { AuthModal } from './components/AuthModal';
import { TradeSignal, WatchlistItem, UserProfile } from './types';

export function App() {
  // Navigation
  const [activeTab, setActiveTab] = useState<'landing' | 'terminal' | 'history' | 'watchlist' | 'admin'>('landing');

  // User & Auth Token State
  const [authToken, setAuthToken] = useState<string | null>(() => {
    try {
      return localStorage.getItem('tradeguard_token');
    } catch {
      return null;
    }
  });

  const [user, setUser] = useState<UserProfile | null>(() => {
    try {
      const savedToken = localStorage.getItem('tradeguard_token');
      const savedUser = localStorage.getItem('tradeguard_user');
      if (savedToken && savedUser) {
        return JSON.parse(savedUser);
      }
    } catch {
      // ignore
    }
    return null;
  });

  const [authModalOpen, setAuthModalOpen] = useState(false);

  // Signals State
  const [signals, setSignals] = useState<TradeSignal[]>([]);

  // Watchlist State
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);

  // Selected Signal for Inspection
  const [inspectedSignal, setInspectedSignal] = useState<TradeSignal | null>(null);

  // Verify stored token or obtain signed sandbox trader session from server
  useEffect(() => {
    let cancelled = false;
    async function initAuthSession() {
      const currentToken = localStorage.getItem('tradeguard_token');
      if (currentToken) {
        try {
          const res = await fetch('/api/auth/session', {
            headers: { Authorization: `Bearer ${currentToken}` },
          });
          if (res.ok) {
            const data = await res.json();
            if (!cancelled && data.success && data.user) {
              setUser(data.user);
              setAuthToken(currentToken);
              localStorage.setItem('tradeguard_user', JSON.stringify(data.user));
              return;
            }
          }
        } catch {
          // fallback below
        }
        // Invalid/expired token: clear it
        localStorage.removeItem('tradeguard_token');
        localStorage.removeItem('tradeguard_user');
      }

      // Bootstrap default non-admin sandbox trader session via server-signed token
      try {
        const demoRes = await fetch('/api/auth/demo', { method: 'POST' });
        if (demoRes.ok) {
          const demoData = await demoRes.json();
          if (!cancelled && demoData.success && demoData.token && demoData.user) {
            setAuthToken(demoData.token);
            setUser(demoData.user);
            localStorage.setItem('tradeguard_token', demoData.token);
            localStorage.setItem('tradeguard_user', JSON.stringify(demoData.user));
          }
        }
      } catch {
        // user can sign in manually via AuthModal
      }
    }
    initAuthSession();
    return () => {
      cancelled = true;
    };
  }, []);

  const getAuthHeaders = (withJson = false): Record<string, string> => {
    const headers: Record<string, string> = {};
    if (withJson) headers['Content-Type'] = 'application/json';
    const token = authToken || localStorage.getItem('tradeguard_token');
    if (token) headers['Authorization'] = `Bearer ${token}`;
    return headers;
  };

  // Fetch initial data from server API
  const fetchSignals = async () => {
    const token = authToken || localStorage.getItem('tradeguard_token');
    if (!token) return;
    try {
      const res = await fetch('/api/signals', {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) return;
      const data = await res.json();
      if (data.success && data.signals) {
        setSignals(data.signals);
      }
    } catch (e) {
      console.error('Failed to load signals', e);
    }
  };

  const fetchWatchlist = async () => {
    const token = authToken || localStorage.getItem('tradeguard_token');
    if (!token) return;
    try {
      const res = await fetch('/api/watchlist', {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) return;
      const data = await res.json();
      if (data.success && data.watchlist) {
        setWatchlist(data.watchlist);
      }
    } catch (e) {
      console.error('Failed to load watchlist', e);
    }
  };

  useEffect(() => {
    if (authToken) {
      fetchSignals();
      fetchWatchlist();
    } else {
      setSignals([]);
      setWatchlist([]);
    }
  }, [authToken]);

  // Auth Handlers
  const handleAuthSuccess = (u: UserProfile, token?: string) => {
    setUser(u);
    localStorage.setItem('tradeguard_user', JSON.stringify(u));
    if (token) {
      setAuthToken(token);
      localStorage.setItem('tradeguard_token', token);
    }
  };

  const handleSignOut = () => {
    setUser(null);
    setAuthToken(null);
    setSignals([]);
    setWatchlist([]);
    localStorage.removeItem('tradeguard_user');
    localStorage.removeItem('tradeguard_token');
    if (activeTab === 'admin') {
      setActiveTab('landing');
    }
  };

  // Watchlist Handlers
  const isWatchlisted = (signalId: string) => {
    return watchlist.some(w => w.signal_id === signalId);
  };

  const handleSaveToWatchlist = async (signal: TradeSignal) => {
    if (isWatchlisted(signal.id)) return;
    if (!authToken) {
      setAuthModalOpen(true);
      return;
    }

    try {
      const res = await fetch('/api/watchlist', {
        method: 'POST',
        headers: getAuthHeaders(true),
        body: JSON.stringify({
          signalId: signal.id,
          notes: `Setup identified on ${signal.timeframe} via ${signal.strategy}`,
        }),
      });
      const data = await res.json();
      if (data.success) {
        fetchWatchlist();
      }
    } catch (e) {
      console.error('Failed to add to watchlist', e);
    }
  };

  const handleRemoveFromWatchlist = async (id: string) => {
    try {
      const res = await fetch(`/api/watchlist/${id}`, {
        method: 'DELETE',
        headers: getAuthHeaders(),
      });
      if (res.ok) {
        setWatchlist(prev => prev.filter(w => w.id !== id));
      }
    } catch (e) {
      console.error('Failed to remove from watchlist', e);
    }
  };

  const handleUpdateWatchlistStatus = async (
    id: string,
    status: 'ACTIVE' | 'TRIGGERED' | 'INVALIDATED' | 'CLOSED'
  ) => {
    try {
      const res = await fetch(`/api/watchlist/${id}`, {
        method: 'PATCH',
        headers: getAuthHeaders(true),
        body: JSON.stringify({ status }),
      });
      if (res.ok) {
        setWatchlist(prev =>
          prev.map(w => (w.id === id ? { ...w, status } : w))
        );
      }
    } catch (e) {
      console.error('Failed to update status', e);
    }
  };

  const handleUpdateWatchlistNotes = async (id: string, notes: string) => {
    try {
      const res = await fetch(`/api/watchlist/${id}`, {
        method: 'PATCH',
        headers: getAuthHeaders(true),
        body: JSON.stringify({ notes }),
      });
      if (res.ok) {
        setWatchlist(prev =>
          prev.map(w => (w.id === id ? { ...w, notes } : w))
        );
      }
    } catch (e) {
      console.error('Failed to update notes', e);
    }
  };

  // Signal History Handlers
  const handleUpdateSignalOutcome = async (
    signalId: string,
    outcome: 'WIN' | 'LOSS' | 'BREAKEVEN' | 'PENDING',
    realizedR?: number
  ) => {
    try {
      const res = await fetch(`/api/signals/${signalId}/outcome`, {
        method: 'PATCH',
        headers: getAuthHeaders(true),
        body: JSON.stringify({ outcome, realizedR }),
      });
      if (res.ok) {
        setSignals(prev =>
          prev.map(s => (s.id === signalId ? { ...s, outcome, realizedR } : s))
        );
      }
    } catch (e) {
      console.error('Failed to update outcome', e);
    }
  };

  const handleDeleteSignal = async (signalId: string) => {
    try {
      const res = await fetch(`/api/signals/${signalId}`, {
        method: 'DELETE',
        headers: getAuthHeaders(),
      });
      if (res.ok) {
        setSignals(prev => prev.filter(s => s.id !== signalId));
      }
    } catch (e) {
      console.error('Failed to delete signal', e);
    }
  };

  const handleSelectSignalToInspect = (sig: TradeSignal) => {
    setInspectedSignal(sig);
    setActiveTab('terminal');
  };

  const handleNewSignalGenerated = (sig: TradeSignal) => {
    setSignals(prev => [sig, ...prev]);
  };

  return (
    <div className="min-h-screen bg-[#0b0e14] text-slate-100 font-sans flex flex-col selection:bg-emerald-500/30 selection:text-emerald-200 w-full max-w-full overflow-x-clip">
      {/* Navigation Bar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        user={user}
        onOpenAuth={() => setAuthModalOpen(true)}
        onSignOut={handleSignOut}
        watchlistCount={watchlist.length}
        historyCount={signals.length}
      />

      {/* Main View Router */}
      <main className="flex-1 w-full max-w-full overflow-x-clip">
        {activeTab === 'landing' && (
          <LandingPage
            onOpenTerminal={() => setActiveTab('terminal')}
            onOpenAuth={() => setAuthModalOpen(true)}
          />
        )}

        {activeTab === 'terminal' && (
          <Terminal
            user={user}
            authToken={authToken}
            inspectedSignal={inspectedSignal}
            onOpenAuth={() => setAuthModalOpen(true)}
            onSaveToWatchlist={handleSaveToWatchlist}
            isWatchlisted={isWatchlisted}
            onSignalGenerated={handleNewSignalGenerated}
          />
        )}

        {activeTab === 'history' && (
          <SignalHistory
            signals={signals}
            onSelectSignal={handleSelectSignalToInspect}
            onUpdateOutcome={handleUpdateSignalOutcome}
            onSaveToWatchlist={handleSaveToWatchlist}
            isWatchlisted={isWatchlisted}
            onDeleteSignal={handleDeleteSignal}
          />
        )}

        {activeTab === 'watchlist' && (
          <Watchlist
            items={watchlist}
            onOpenSignalInTerminal={handleSelectSignalToInspect}
            onRemoveItem={handleRemoveFromWatchlist}
            onUpdateStatus={handleUpdateWatchlistStatus}
            onUpdateNotes={handleUpdateWatchlistNotes}
          />
        )}

        {activeTab === 'admin' && <AdminLogs user={user} authToken={authToken} />}
      </main>

      {/* Auth Modal */}
      <AuthModal
        isOpen={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        onSuccess={handleAuthSuccess}
      />
    </div>
  );
}
export default App;
