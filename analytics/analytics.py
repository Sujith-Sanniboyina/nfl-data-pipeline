import pandas as pd
import logging
from pathlib import Path
from datetime import datetime

from load.load import NFLDataLoader

logger = logging.getLogger(__name__)


# Generates text reports from the NFL database
class NFLReportGenerator:

    def __init__(self, db_path="data/nfl_data.db"):
        self.loader = NFLDataLoader(db_path)
        self.db_path = db_path

    # Builds a top 10 teams report for a given season, saves it to a file, and prints it
    def generate_top_teams_report(self, season, output_file="top_teams_report.txt"):
        try:
            self.loader.connect()

            top_teams = self.loader.get_top_teams(season, limit=10)

            if top_teams.empty:
                return "No data available for the specified season."

            report_lines = []
            report_lines.append(f"NFL Top Teams Report - {season} Season")
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            report_lines.append("Top 10 Teams by Win Percentage:")
            report_lines.append("-" * 60)

            for idx, row in top_teams.iterrows():
                report_lines.append(
                    f"{idx+1:2d}. {row['team']:3s} | "
                    f"Record: {int(row['wins'])}-{int(row['losses'])} | "
                    f"Win%: {row['win_percentage']:.3f} | "
                    f"Points: {int(row['total_points'])} | "
                    f"Points Allowed: {int(row['total_points_allowed'])}"
                )

            report_content = '\n'.join(report_lines)

            output_path = Path(output_file)
            with open(output_path, 'w') as f:
                f.write(report_content)

            logger.info(f"Report saved to {output_path}")
            print(report_content)

            return str(output_path)

        finally:
            self.loader.disconnect()

    # Pulls stats for a specific team and season and returns a formatted summary string
    def generate_team_summary(self, team, season):
        try:
            self.loader.connect()

            team_stats = self.loader.get_team_stats(team, season)

            if team_stats.empty:
                return f"No data found for {team} in {season}"

            row = team_stats.iloc[0]

            summary = f"""
Team Summary: {team} - {season} Season
{'=' * 40}
Record: {int(row['wins'])}-{int(row['losses'])}
Win Percentage: {row['win_percentage']:.3f}

Offense:
- Points Scored:  {int(row['total_points'])} ({row['avg_points']:.1f} per game)
- Total Yards:    {int(row['total_yards'])} ({row['avg_yards']:.1f} per game)
- Touchdowns:     {int(row['total_touchdowns'])} ({row['avg_touchdowns']:.1f} per game)
- Turnovers:      {int(row['total_turnovers'])} ({row['avg_turnovers']:.1f} per game)

Defense:
- Points Allowed: {int(row['total_points_allowed'])} ({row['avg_points_allowed']:.1f} per game)
"""
            return summary

        finally:
            self.loader.disconnect()


def main():
    reporter = NFLReportGenerator()

    reporter.generate_top_teams_report(2023)
    print(reporter.generate_team_summary('KC', 2023))


if __name__ == "__main__":
    main()