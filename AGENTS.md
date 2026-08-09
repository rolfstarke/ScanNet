### ScanNet
- first priority: https://github.com/ScanNet/ScanNet tools and utils.
- second priority: the prediction model repositories tools and utils.
- third priority: ov3discomp tools and utils.
- all your code should go to spellbook/
- always stick to the datastructure of /data/scannet
- always make sure to keep ScanNet fully intact and operable with all additions

### memory externalization
- short term memory: write @scout reports to spellbook/tmp
- short term memory: write detailed action plans for @build agents directly to separate .md files in spellbook/tmp. summarize in chat.
- tailor the action plans to deepseek v4 flash
- wipe reports and action plans and reports after build.
- long term memory: spellbook/PROJECT_STATUS.md — repository structure, project goal, and project plan for new sessions; never use it as a debug or activity log
- bugs and problems: GitHub Issues (rolfstarke/ScanNet); one issue per problem, log concise theoretical reasoning and attempted fixes as issue comments, close only after verified resolution, reference issues as #N
- before /compact: update the relevant issues first; new sessions read PROJECT_STATUS.md, then the relevant issues
- use subagent as much as possible to save usage






