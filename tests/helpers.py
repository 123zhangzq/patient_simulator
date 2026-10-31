"""Small authored profiles for behavioral tests; no source files required."""


def profile_data(participant_id="TEST001"):
    return {
        "schema_version": 1, "participant_id": participant_id, "role": "stroke_survivor",
        "assessment_date": "2022-10-01", "simulation_date": "2022-12-01",
        "background": {"age": 62, "language": "English", "education_years": 12},
        "assessments": {
            "moca": {"total": 22, "attention": 4, "language": 2, "delayed_recall": 1,
                     "category_cue": 3, "multiple_choice_cue": 1, "visuospatial_executive": 4,
                     "abstraction": 2, "orientation": 6, "naming": 3},
            "sdmt": {"total": 30},
            "dass21": {"depression": 2, "anxiety": 0, "stress": 4},
        },
        "known_facts": [{"id": "home", "text": "I live with my spouse.", "source": "PRIVATE_SOURCE_SENTINEL"}],
        "initial_memories": [], "provenance": {"source": "PRIVATE_SOURCE_SENTINEL"},
        "unknowns": ["Previous task attendance and homework are unknown."],
    }
