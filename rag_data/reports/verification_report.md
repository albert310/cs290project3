# Clean RAG Verification Report

- Database: `/2022533109/chenyuhan/cs290project3/rag_data/db/shanghaitech_sist.sqlite`
- Cases: **7**
- Passed: **7/7**

| id | ok | matched_terms | top_tier | top_category | top_title |
| --- | --- | --- | --- | --- | --- |
| fact_university_jointly_founded | True | Shanghai Municipal Government; Chinese Academy of Sciences | verified_seed | university_overview | 上海科技大学是谁共同举办和建设的？ |
| fact_university_established_2013 | True | 2013; ShanghaiTech University | verified_seed | university_overview | 上海科技大学正式建立是哪一年？ |
| fact_university_address | True | 393; Huaxia; Pudong | verified_seed | university_contact | 上海科技大学浦东校区地址是什么？ |
| fact_sist_full_name | True | School of Information Science and Technology; ShanghaiTech | verified_seed | sist_overview | SIST 的英文全称是什么？ |
| fact_sist_first_school | True | first; School of Information Science and Technology | verified_seed | sist_overview | 信息科学与技术学院在上海科技大学建校初期处于什么地位？ |
| fact_sist_courses_2024_2025 | True | SIST; 2024-2025; Courses | verified_seed | sist_courses | 哪里可以查到信息学院 2024-2025 学年课程？ |
| fact_wang_haoyu_email | True | 王浩宇; wanghy@shanghaitech.edu.cn | verified_seed | sist_faculty | 王浩宇教授的邮箱是什么？ |

## Details

### fact_university_jointly_founded PASS

- Question: 上海科技大学是谁共同举办和建设的？
- Expected terms: Shanghai Municipal Government; Chinese Academy of Sciences
- Matched terms: Shanghai Municipal Government; Chinese Academy of Sciences
- Top URL: https://www.shanghaitech.edu.cn/en/997/main.psp
- Top snippet: 上海科技大学是谁共同举办和建设的?
标准答案/核验事实: ShanghaiTech University is jointly founded by Shanghai Municipal Government and Chinese Academy of Sciences.
关键核验词: Shanghai Municipal Government、Chinese Academy of Sciences
来源说明: Official ShanghaiTech About page.
来源 URL: https://www.shanghaitech.edu.cn/en/997/main.psp

### fact_university_established_2013 PASS

- Question: 上海科技大学正式建立是哪一年？
- Expected terms: 2013; ShanghaiTech University
- Matched terms: 2013; ShanghaiTech University
- Top URL: https://www.shanghaitech.edu.cn/en/997/main.psp
- Top snippet: 上海科技大学正式建立是哪一年?
标准答案/核验事实: ShanghaiTech University was officially established in 2013.
关键核验词: 2013、ShanghaiTech University
来源说明: Official ShanghaiTech About page.
来源 URL: https://www.shanghaitech.edu.cn/en/997/main.psp

### fact_university_address PASS

- Question: 上海科技大学浦东校区地址是什么？
- Expected terms: 393; Huaxia; Pudong
- Matched terms: 393; Huaxia; Pudong
- Top URL: https://www.shanghaitech.edu.cn/en/1059/list.psp
- Top snippet: 上海科技大学浦东校区地址是什么?
标准答案/核验事实: The Pudong campus address is 393 Huaxia Middle Road, Pudong New Area, Shanghai.
关键核验词: 393、Huaxia、Pudong
来源说明: Official ShanghaiTech contact page.
来源 URL: https://www.shanghaitech.edu.cn/en/1059/list.psp

### fact_sist_full_name PASS

- Question: SIST 的英文全称是什么？
- Expected terms: School of Information Science and Technology; ShanghaiTech
- Matched terms: School of Information Science and Technology; ShanghaiTech
- Top URL: https://sist.shanghaitech.edu.cn/sist_en/
- Top snippet: SIST 的英文全称是什么?
标准答案/核验事实: SIST stands for School of Information Science and Technology at ShanghaiTech University.
关键核验词: School of Information Science and Technology、ShanghaiTech
来源说明: Official SIST English site.
来源 URL: https://sist.shanghaitech.edu.cn/sist_en/

### fact_sist_first_school PASS

- Question: 信息科学与技术学院在上海科技大学建校初期处于什么地位？
- Expected terms: first; School of Information Science and Technology
- Matched terms: first; School of Information Science and Technology
- Top URL: https://sist.shanghaitech.edu.cn/sist_en/
- Top snippet: 信息科学与技术学院在上海科技大学建校初期处于什么地位?
标准答案/核验事实: SIST is described as the first school established at ShanghaiTech University.
关键核验词: first、School of Information Science and Technology
来源说明: Official SIST English site.
来源 URL: https://sist.shanghaitech.edu.cn/sist_en/

### fact_sist_courses_2024_2025 PASS

- Question: 哪里可以查到信息学院 2024-2025 学年课程？
- Expected terms: SIST; 2024-2025; Courses
- Matched terms: SIST; 2024-2025; Courses
- Top URL: https://faculty.sist.shanghaitech.edu.cn/office/Academics/Courses/SIST_2024-2025_Courses.htm
- Top snippet: 哪里可以查到信息学院 2024-2025 学年课程?
标准答案/核验事实: The official SIST 2024-2025 course list is published by the SIST faculty academics site.
关键核验词: SIST、2024-2025、Courses
来源说明: Official SIST course list.
来源 URL: https://faculty.sist.shanghaitech.edu.cn/office/Academics/Courses/SIST_2024-2025_Courses.htm

### fact_wang_haoyu_email PASS

- Question: 王浩宇教授的邮箱是什么？
- Expected terms: 王浩宇; wanghy@shanghaitech.edu.cn
- Matched terms: 王浩宇; wanghy@shanghaitech.edu.cn
- Top URL: https://sist.shanghaitech.edu.cn/wanghy/main.htm
- Top snippet: 王浩宇教授的邮箱是什么?
标准答案/核验事实: Wang Haoyu's official SIST profile lists wanghy@shanghaitech.edu.cn.
关键核验词: 王浩宇、wanghy@shanghaitech.edu.cn
来源说明: Official SIST faculty profile.
来源 URL: https://sist.shanghaitech.edu.cn/wanghy/main.htm
