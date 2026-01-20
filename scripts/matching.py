#!/usr/bin/env python3
"""
Demo script for resume-job matching system.
Work in progress - basic terminal output for testing.
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_resume(resume_path):
    """Parse a resume file and return results."""
    from src.ml.nlp import get_resume_parser

    parser = get_resume_parser()
    result = parser.parse_file(resume_path)

    if not result.success:
        print(f"  [ERROR] Failed to parse: {result.errors}")
        return None

    # extract name
    name = "Unknown"
    if result.contact:
        name = result.contact.get("full_name") or f"{result.contact.get('first_name', '')} {result.contact.get('last_name', '')}".strip()
        if not name or name == " ":
            name = result.contact.get("email", "Unknown").split("@")[0] if result.contact.get("email") else "Unknown"

    print(f"  Name: {name}")
    print(f"  Email: {result.contact.get('email', 'N/A') if result.contact else 'N/A'}")
    print(f"  Skills found: {len(result.skills)}")
    if result.skills:
        skills_list = [s["name"] for s in result.skills[:8]]
        print(f"  Top skills: {', '.join(skills_list)}")
    print(f"  Experience: {result.total_experience_years:.1f} years")
    print(f"  Education: {result.highest_education or 'N/A'}")
    print(f"  Confidence: {result.overall_confidence:.0%}")

    return result


def parse_jd(jd_path):
    """Parse a job description file."""
    from src.ml.nlp import get_jd_parser

    parser = get_jd_parser()
    result = parser.parse_file(jd_path)

    print(f"  Title: {result.title}")
    print(f"  Company: {result.company_name}")
    print(f"  Type: {result.employment_type}")
    print(f"  Level: {result.experience_level}")
    if result.experience_years_min:
        print(f"  Exp required: {result.experience_years_min}+ years")
    print(f"  Required skills ({len(result.required_skills)}): {', '.join(result.required_skills[:6])}")
    if result.preferred_skills:
        print(f"  Preferred skills: {', '.join(result.preferred_skills[:4])}")

    return result


def match_candidate(resume_result, jd_result):
    """Match a candidate against a job."""
    from src.core.matching import get_matching_engine

    engine = get_matching_engine()
    result = engine.match(resume_result, jd_result)
    return result


def run_batch(jds_dir, resumes_dir):
    """Run batch matching - all resumes against all JDs."""
    from src.ml.nlp import get_resume_parser, get_jd_parser
    from src.core.matching import get_matching_engine

    resume_parser = get_resume_parser()
    jd_parser = get_jd_parser()
    engine = get_matching_engine()

    # get all files
    jd_files = list(jds_dir.glob("*.pdf"))
    resume_files = list(resumes_dir.glob("*.pdf"))

    print(f"\nFound {len(jd_files)} job descriptions and {len(resume_files)} resumes")

    # parse all resumes once
    print("\nParsing resumes...")
    parsed_resumes = []
    for i, resume_path in enumerate(resume_files, 1):
        print(f"  [{i}/{len(resume_files)}] {resume_path.name[:45]}...", end="")
        try:
            result = resume_parser.parse_file(resume_path)
            if result.success:
                parsed_resumes.append((resume_path.name, result))
                print(" OK")
            else:
                print(" FAILED")
        except Exception as e:
            print(f" ERROR")

    print(f"\nSuccessfully parsed {len(parsed_resumes)} resumes")

    # process each JD
    for jd_idx, jd_path in enumerate(jd_files, 1):
        print("\n" + "="*60)
        print(f"JOB {jd_idx}/{len(jd_files)}: {jd_path.name}")
        print("="*60)

        jd_result = jd_parser.parse_file(jd_path)
        print(f"Position: {jd_result.title}")
        print(f"Skills required: {', '.join(jd_result.required_skills[:5])}")

        # match all resumes against this JD
        results = []
        for resume_name, resume_result in parsed_resumes:
            match = engine.match(resume_result, jd_result)
            match.resume_file = resume_name
            results.append(match)

        # rank
        ranked = engine.rank_candidates(results)

        # display top 10
        print(f"\n{'Rank':<5} {'Candidate':<25} {'Score':<8} {'Level':<10} {'Skills'}")
        print("-"*55)

        for i, m in enumerate(ranked[:10], 1):
            skills_info = f"{len(m.matched_skills)}/{len(m.skill_matches)}"
            print(f"{i:<5} {m.candidate_name[:24]:<25} {m.overall_score:.0%}     {m.score_level.value:<10} {skills_info}")

        if len(ranked) > 10:
            print(f"  ... and {len(ranked) - 10} more candidates")

        # summary for this JD
        excellent = sum(1 for m in ranked if m.overall_score >= 0.85)
        good = sum(1 for m in ranked if 0.70 <= m.overall_score < 0.85)
        fair = sum(1 for m in ranked if 0.50 <= m.overall_score < 0.70)

        print(f"\nSummary: {excellent} excellent, {good} good, {fair} fair")

        if ranked:
            print(f"Top match: {ranked[0].candidate_name} ({ranked[0].overall_score:.0%})")

    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"Processed {len(jd_files)} JDs against {len(parsed_resumes)} resumes")


def main():
    parser = argparse.ArgumentParser(description="Resume-Job Matching Demo")
    parser.add_argument("--resume", type=Path, help="Resume file path")
    parser.add_argument("--jd", type=Path, help="Job description file path")
    parser.add_argument("--batch", action="store_true", help="Batch mode")
    parser.add_argument("--resumes-dir", type=Path,
                       default=project_root / "data" / "raw" / "resumes")
    parser.add_argument("--jds-dir", type=Path,
                       default=project_root / "data" / "raw" / "job_descriptions")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("AI-ATS: Resume-Job Matching System")
    print("="*60)

    try:
        if args.batch or (not args.resume and not args.jd):
            # batch mode - all JDs against all resumes
            run_batch(args.jds_dir, args.resumes_dir)

        elif args.resume and args.jd:
            # single match
            print("\n--- RESUME PARSING ---")
            resume_result = parse_resume(args.resume)

            if resume_result:
                print("\n--- JD PARSING ---")
                jd_result = parse_jd(args.jd)

                print("\n--- MATCHING ---")
                match = match_candidate(resume_result, jd_result)

                print(f"\nOverall Score: {match.overall_score:.0%} ({match.score_level.value})")
                print(f"\nBreakdown:")
                print(f"  Skills:     {match.skills_score:.0%} (weight: 35%)")
                print(f"  Experience: {match.experience_score:.0%} (weight: 25%)")
                print(f"  Education:  {match.education_score:.0%} (weight: 15%)")
                print(f"  Keywords:   {match.keyword_score:.0%} (weight: 25%)")

                print(f"\nMatched skills: {', '.join(match.matched_skills[:6])}")
                if match.missing_skills:
                    print(f"Missing skills: {', '.join(match.missing_skills[:6])}")

                if match.explanation:
                    print(f"\nSummary: {match.explanation.summary}")
        else:
            print("Usage: python demo_matching.py --batch")
            print("       python demo_matching.py --resume <file> --jd <file>")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
