# Project Site Template

Github Pages template for project landing page.

*This template is based on the [Cayman theme](https://github.com/pages-themes/cayman) template, a Jekyll theme for GitHub Pages.*

## 🚀 Setup
> **Note**: This site requires [Ruby](https://www.ruby-lang.org/). To check if you have Ruby already installed, run `ruby -v` in your terminal. If it is not, follow these [instructions](https://www.ruby-lang.org/en/documentation/installation/).

To preview the theme locally and populate the template with your project information:
1. Clone this repository
```
git clone https://github.com/princeton-nlp/project-page-template
```
2. `cd` into the theme's directory
3. Run `./setup` to install the necessary dependencies
4. Run `bundle exec jekyll serve` to start the preview server
5. Visit [`localhost:4000`](http://localhost:4000) in your browser to preview the theme

## 🛠️ Usage
Updating this template to describe your project can be done in three files:
* The **contents** of the webpage can be edited in the `index.md` file.
* The **layout** of the webpage can be modified in the `_layouts/default.html` file.
* The **header** + **metadata** of the webpage can be configured in the `_config.yml` file.

> **Tip**: Check out an example - the DataMUX project [page](https://princeton-nlp.github.io/DataMUX/) + [code](https://github.com/princeton-nlp/DataMUX/tree/website)!

To maintain consistency in styling across all project sites, modifying the CSS is discouraged. With that said, if you would like to, you can modify styling within the `/_sass` folder. The `variables.scss` variables defines the colors used for the webpage entities (i.e. headers, text, code block coloring), and these variables are used in the `pnlp-theme.scss` file. The other `.scss` files not mentioned here control the appearance + organization of elements of the page.

## 🚢 Deployment
Once you run `bundle exec jekyll serve`, a `_site` folder is created, which contains the generated HTML + CSS code that is your website. To deploy this code either anonymously or publicly, follow one of the set of deployment steps below.

### 👋 Public Release
If you are deploying *publicly*, it is recommended to deploy your site under the `princeton-nlp.github.io` domain (i.e. `princeton-nlp.github.io/DataMUX`). To do so, please do the following:
* Push your desired changes to the GitHub repository for your project page
* Click "Settings" to navigate to the repository settings
* In the "Code and automation" section of the sidebar, click "Pages"
* Under **Source**, select the dropdown menu to choose which branch to use as the source of your web page
* Select "Save" to build the webpage and host it a the default princeton-nlp.github.io/project-name domain
* If you'd like to use a custom domain name, see the [GitHub Pages Docs](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site) for more help

### 👤 Anonymouse Release
If you are deploying *anonymously*, it is recommended to deploy your site under a `.github.io`  page of an anonymous Github organization. To do so, please do the following:
* Navigate to the [organizations](https://github.com/settings/organizations) page
* Click `New Organization` on the top right
* Select the `Free` tier (click the `Create a free organization` button)
* Fill in the fields to setup the organization. Make sure none of the information you put down directly references Princeton NLP.
* You can fill in or skip through the organization set up information - it is not important
* Click the `Create a new repository` button and name it `<organization name>.github.io` (it must be a public repository for the page to be visible)

You can now deploy your website code to `<organization name>.github.io`. To ensure anonymity, go through the following checklist:
- [ ] Remove the `authors` and `citation` section from `index.md`
- [ ] Change the header gradient colors (`$header-bg-color`, `$header-bg-color-secondary`) and text header color (`$section-heading-color`) in the `_sass/variables.scss` file to a non-Princeton color scheme.
- [ ] Copy + paste the organization URL in an incognito tab. Check that your GitHub account is not publicly associated with the organization

## 📜 Resources
This website template renders Markdown in a HTML + CSS template using [Jekyll](https://jekyllrb.com/), a static site generator written in Ruby. The following are resources that would be helpful for editing the webpage content (`index.md`) or modifying the HTML template (`_layouts/default.html`) for your needs.

* [Markdown Cheat Sheet](https://www.markdownguide.org/basic-syntax/)
* [Jekyll variables documentation](https://jekyllrb.com/docs/variables/)
* [Cayman theme](https://github.com/pages-themes/cayman)
