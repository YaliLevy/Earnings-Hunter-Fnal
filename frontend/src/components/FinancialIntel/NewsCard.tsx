import { motion } from 'framer-motion';
import type { NewsArticle } from '../../types';

interface NewsCardProps {
  articles: NewsArticle[];
}

export function NewsCard({ articles }: NewsCardProps) {
  const displayArticles = articles.slice(0, 6);

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      {/* Header */}
      <h3 className="text-xs tracking-widest text-text-muted mb-4">LATEST NEWS</h3>

      {/* News Items */}
      {displayArticles.length > 0 ? (
        <div className="space-y-1">
          {displayArticles.map((article, index) => (
            <motion.a
              key={index}
              href={article.url || '#'}
              target="_blank"
              rel="noopener noreferrer"
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className="flex items-start gap-2 py-2 border-b border-border last:border-b-0
                         hover:bg-surface-hover rounded transition-colors group"
            >
              {/* Sentiment Dot */}
              <div
                className={`w-1.5 h-1.5 rounded-full mt-1.5 shrink-0 ${
                  article.sentiment === 'bullish'
                    ? 'bg-signal-green'
                    : article.sentiment === 'bearish'
                    ? 'bg-crimson'
                    : 'bg-text-muted'
                }`}
              />

              {/* Title */}
              <span className="text-xs text-text-secondary group-hover:text-signal-green line-clamp-2 transition-colors">
                {article.title}
              </span>
            </motion.a>
          ))}
        </div>
      ) : (
        <div className="text-center py-6 text-text-muted text-sm">No news available</div>
      )}
    </div>
  );
}
