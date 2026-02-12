import { useState, useEffect, useRef } from 'react';
import * as Switch from '@radix-ui/react-switch';
import { Zap, Search } from 'lucide-react';

interface CommandBarProps {
  deepReasoning: boolean;
  onDeepReasoningChange: (value: boolean) => void;
  onSearch: (symbol: string) => void;
  isLoading: boolean;
}

export function CommandBar({
  deepReasoning,
  onDeepReasoningChange,
  onSearch,
  isLoading,
}: CommandBarProps) {
  const [searchValue, setSearchValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  // Cmd+K shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const symbol = searchValue.trim().toUpperCase();
    if (symbol && !isLoading) {
      onSearch(symbol);
    }
  };

  return (
    <header className="h-14 bg-bg-dark/95 backdrop-blur-xl border-b border-border
                       flex items-center justify-between px-6 sticky top-0 z-40">
      {/* Logo */}
      <div className="flex items-center gap-2">
        <Zap className="w-5 h-5 text-signal-green" />
        <span className="text-sm font-bold tracking-[0.2em] text-text-primary">
          EARNINGS HUNTER
        </span>
      </div>

      {/* Search */}
      <form onSubmit={handleSubmit} className="flex-1 max-w-md mx-8">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
          <input
            ref={inputRef}
            type="text"
            value={searchValue}
            onChange={(e) => setSearchValue(e.target.value.toUpperCase())}
            placeholder="Search ticker... ⌘K"
            disabled={isLoading}
            className="w-full bg-surface border border-border rounded-lg pl-10 pr-4 py-2.5
                       text-sm text-center text-text-primary placeholder:text-text-muted
                       focus:border-signal-green focus:outline-none transition-colors
                       disabled:opacity-50"
          />
        </div>
      </form>

      {/* Right Controls */}
      <div className="flex items-center gap-6">
        {/* Deep Reasoning Toggle */}
        <div className="flex items-center gap-3">
          <Switch.Root
            checked={deepReasoning}
            onCheckedChange={onDeepReasoningChange}
            className="w-9 h-5 bg-border rounded-full relative data-[state=checked]:bg-purple
                       transition-colors cursor-pointer"
          >
            <Switch.Thumb
              className="block w-4 h-4 bg-white rounded-full transition-transform
                         translate-x-0.5 data-[state=checked]:translate-x-[18px]"
            />
          </Switch.Root>
          <span
            className={`text-xs font-medium tracking-wider transition-colors ${
              deepReasoning ? 'text-purple' : 'text-text-muted'
            }`}
          >
            DEEP REASONING
          </span>
        </div>

        {/* Live Indicator */}
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-signal-green animate-live-pulse" />
          <span className="text-xs font-semibold tracking-wider text-signal-green">
            LIVE
          </span>
        </div>
      </div>
    </header>
  );
}
