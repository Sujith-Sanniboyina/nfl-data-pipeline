--
-- NFL Data Pipeline Database Schema
-- Portable across SQLite (local tests) and PostgreSQL (Supabase).
-- No SERIAL/AUTOINCREMENT id column -- each DB's implicit rowid/ctid is enough,
-- and load.py replaces row contents (DELETE + append) rather than dropping tables,
-- so these CREATE TABLE IF NOT EXISTS statements only really run once per environment.
--
CREATE TABLE IF NOT EXISTS team_season_stats (
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

CREATE TABLE IF NOT EXISTS team_game_stats (
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
    offensive_plays INTEGER,
    defensive_plays INTEGER,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(game_id, team)
);
-- Weekly player-level stats. This is what the prop predictor model trains on --
-- one row per player per week, so rolling averages can be computed per player/stat.
CREATE TABLE IF NOT EXISTS player_game_stats (
    player_id          TEXT    NOT NULL,
    player_name        TEXT    NOT NULL,
    position            TEXT,
    team               TEXT    NOT NULL,
    opponent           TEXT,
    season             INTEGER NOT NULL,
    week               INTEGER NOT NULL,
    home_game          BOOLEAN,
    rushing_yards      INTEGER,
    rushing_attempts   INTEGER,
    receiving_yards    INTEGER,
    receptions         INTEGER,
    targets            INTEGER,
    passing_yards      INTEGER,
    passing_attempts   INTEGER,
    rushing_tds        INTEGER,
    receiving_tds      INTEGER,
    passing_tds        INTEGER,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_id, season, week)
);

--
-- Create indexes for better query performance
--
CREATE INDEX IF NOT EXISTS idx_team_season_stats_season ON team_season_stats(season);
CREATE INDEX IF NOT EXISTS idx_team_season_stats_team   ON team_season_stats(team);
CREATE INDEX IF NOT EXISTS idx_team_game_stats_game_id  ON team_game_stats(game_id);
CREATE INDEX IF NOT EXISTS idx_team_game_stats_team     ON team_game_stats(team);
CREATE INDEX IF NOT EXISTS idx_player_game_stats_player ON player_game_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_player_game_stats_season_week ON player_game_stats(season, week);