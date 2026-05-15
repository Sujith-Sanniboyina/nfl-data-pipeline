--
-- NFL Data Pipeline Database Schema
-- SQLite compatible
--
-- Dropping existing tables first
--
DROP TABLE IF EXISTS team_season_stats;
DROP TABLE IF EXISTS team_game_stats;
DROP TABLE IF EXISTS raw_play;
--
-- Adding each table
--
CREATE TABLE team_season_stats (
    id               SERIAL PRIMARY KEY,
    season           INTEGER NOT NULL,
    team             TEXT    NOT NULL,
    total_points     INTEGER,
    avg_points       REAL,
    total_points_allowed INTEGER,
    avg_points_allowed   REAL,
    total_yards      INTEGER,
    avg_yards        REAL,
    total_turnovers  INTEGER,
    avg_turnovers    REAL,
    total_touchdowns INTEGER,
    avg_touchdowns   REAL,
    total_plays      INTEGER,
    avg_plays        REAL,
    wins             INTEGER,
    losses           INTEGER,
    win_percentage   REAL,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(season, team)
);

CREATE TABLE team_game_stats (
    id             SERIAL PRIMARY KEY,
    game_id        TEXT    NOT NULL,
    season         INTEGER,
    week           INTEGER,
    team           TEXT    NOT NULL,
    opponent       TEXT    NOT NULL,
    home_game      BOOLEAN,
    points_scored  INTEGER,
    points_allowed INTEGER,
    total_yards    INTEGER,
    turnovers      INTEGER,
    touchdowns     INTEGER,
    total_plays    INTEGER,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(game_id, team)
);
--
-- Create indexes for better query performance
--
CREATE INDEX IF NOT EXISTS idx_team_season_stats_season ON team_season_stats(season);
CREATE INDEX IF NOT EXISTS idx_team_season_stats_team   ON team_season_stats(team);
CREATE INDEX IF NOT EXISTS idx_team_game_stats_game_id  ON team_game_stats(game_id);
CREATE INDEX IF NOT EXISTS idx_team_game_stats_team     ON team_game_stats(team);