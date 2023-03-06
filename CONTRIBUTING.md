# Contributing to ExaDG
We are happy that you wish to contribute. 
Below, you find information explaining how to contribute to ExaDG.

## Getting started
Please read our [README.md](https://github.com/exadg/exadg/#readme) for general information on ExaDG and how we want to maintain the project.
In [our wiki](https://github.com/exadg/exadg/wiki), you can find out how to install ExaDG and setup your development environment.

## Working with issues
For any ideas, questions, or points that you want to discuss, please create an issue on GitHub.
Typical topics for an issue are for example _bug reports_ and _enhancements_.
Issues enable us to communicate and work effectively as a community.
It is best if you create an issue _before_ working on a new task. 
This will make sure that your efforts are in the best interest of everyone and allow you to work more efficiently.
Provide the necessary information others need to easily understand what the issue is dealing with.
If the issue is related to other issues or PRs in any way, you can also mention these.
Some keywords that describe these relations are _part of_, _follows_, _blocks_, _blocked by_, _closes_, etc. 

## Making changes
To make changes to ExaDG, we use the Fork-and-Branch workflow.
If you are not familiar with this concept, you can find a nice explanation [here](https://blog.scottlowe.org/2015/01/27/using-fork-branch-git-workflow/).
At the end, you will open a pull request (PR), which will give a chance for others to look at and review your changes.
Before you open a PR, please make sure that
 1. your changes adhere to [our coding conventions](https://github.com/exadg/exadg/wiki/Coding-conventions), including indentation rules,
 2. there are no build errors or warnings,
 3. all tests pass, see [running tests with ExaDG](https://github.com/exadg/exadg/wiki/Running-tests),
 4. you added proper doxygen documentation,
 5. your commit messages are descriptive. 
   
We would also appreciate if you write tests for the code part that you want to contribute.
This will not only make sure that your implementation is correct now, but also make sure that future changes by you or other contributors do not break the code.

### PRs and the Review Process
When opening a PR, you should write a descriptive message on the changes that it proposes.
A PR on GitHub can either be in a draft state or ready for review.
Once you mark your PR ready, other community members will review your changes and might ask questions and call for modifications.
Please remember that everybody wants the best for the project, so feel free to express your own ideas and to respect those of the reviewers.
Feel free to assign reviewers you think are most familiar with the relevant modified code and the topic.
While PRs can only be merged by the principal developers (a full and up-to-date list is given [here](https://github.com/exadg/exadg/#authors)), everyone is invited to review code and contribute to the discussions.
Once you address changes requested by code reviewers, the reviewers will check your changes again and finally resolve open conversations.
We also want to encourage rebasing your branch onto the latest master commit before the PR is merged. 