# Clean ShanghaiTech/SIST RAG Database Report

- Built at UTC: `2026-06-04T16:20:11Z`
- Database: `/2022533109/chenyuhan/cs290project3/rag_data/db/shanghaitech_sist.sqlite`
- Documents inserted: **2069**
- Chunks inserted: **10025**
- Duplicate documents skipped: **232**
- Chunks without URL: **0**

## Source Tiers

| source_tier | chunks | avg_quality |
| --- | ---: | ---: |
| local_official_mirror | 9905 | 0.941 |
| live_official | 113 | 0.974 |
| verified_seed | 7 | 0.99 |

## Categories

| category | chunks | avg_quality |
| --- | ---: | ---: |
| sist_faculty | 4975 | 0.958 |
| sist_degree_programs | 2103 | 0.944 |
| sist_news_events | 1045 | 0.905 |
| sist_overview | 677 | 0.905 |
| sist_research | 632 | 0.902 |
| sist_courses | 429 | 0.952 |
| university_overview | 161 | 0.908 |
| university_contact | 3 | 0.91 |

## Top Hosts

| host | chunks |
| --- | ---: |
| faculty.sist.shanghaitech.edu.cn | 5330 |
| sist.shanghaitech.edu.cn | 3584 |
| ssist.shanghaitech.edu.cn | 653 |
| smirc.sist.shanghaitech.edu.cn | 138 |
| ganology.sist.shanghaitech.edu.cn | 98 |
| www.shanghaitech.edu.cn | 89 |
| mpl.sist.shanghaitech.edu.cn | 59 |
| klip-humaco.sist.shanghaitech.edu.cn | 47 |
| pmicc.sist.shanghaitech.edu.cn | 8 |
| cipes.sist.shanghaitech.edu.cn | 6 |
| nice.sist.shanghaitech.edu.cn | 4 |
| vdi.sist.shanghaitech.edu.cn | 3 |
| miblab.sist.shanghaitech.edu.cn | 3 |
| ssc.sist.shanghaitech.edu.cn | 2 |
| summercamp.sist.shanghaitech.edu.cn | 1 |

## Build Summary

```json
{
  "built_at": "2026-06-04T16:20:11Z",
  "chunks_by_category": {
    "sist_courses": 429,
    "sist_degree_programs": 2103,
    "sist_faculty": 4975,
    "sist_news_events": 1045,
    "sist_overview": 677,
    "sist_research": 632,
    "university_contact": 3,
    "university_overview": 161
  },
  "chunks_by_tier": {
    "live_official": 113,
    "local_official_mirror": 9905,
    "verified_seed": 7
  },
  "chunks_inserted": 10025,
  "db_path": "/2022533109/chenyuhan/cs290project3/rag_data/db/shanghaitech_sist.sqlite",
  "docs_seen": 2301,
  "docs_without_chunks": 0,
  "documents_inserted": 2069,
  "duplicate_docs": 232,
  "include_crawl": true,
  "include_local": true,
  "limit_docs": null
}
```
