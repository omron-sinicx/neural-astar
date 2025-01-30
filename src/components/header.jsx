import React from 'react';
import { render } from 'react-dom';
import Authors from '../components/authors.jsx';
import CorporateLogo from '../components/logo.jsx';
import { FaGithub, FaYoutube, FaMedium } from 'react-icons/fa6';
import { FaFilePdf } from 'react-icons/fa';
import { SiArxiv } from 'react-icons/si';
import { Icon } from '@iconify/react';

const HuggingFace = ({ size }) => (
  <Icon icon="logos:hugging-face-icon" style={{ fontSize: size }} />
);

const GoogleColab = ({ size }) => (
  <Icon icon="simple-icons:googlecolab" style={{ fontSize: size }} />
);

class ResourceBtn extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isMobile: window.innerWidth < 600,
    };
    this.icons = {
      paper: FaFilePdf,
      arxiv: SiArxiv,
      code: FaGithub,
      video: FaYoutube,
      blog: FaMedium,
      demo: GoogleColab,
      huggingface: HuggingFace,
    };
    this.handleResize = this.handleResize.bind(this);
  }
  componentDidMount() {
    window.addEventListener('resize', this.handleResize);
    this.handleResize();
  }
  componentWillUnmount() {
    window.removeEventListener('resize', this.handleResize);
  }
  handleResize() {
    this.setState({ isMobile: window.innerWidth < 600 });
  }
  render() {
    if (!this.props.url) return null;
    const aClass = `uk-button uk-button-text uk-padding-remove ${this.props.rid === 0 ? 'uk-first-column' : 'uk-margin-medium-left@s uk-margin-small-left'}`;
    const sClass = 'uk-margin-small-left uk-margin-small-right uk-text-bold';
    const FaIcon = this.icons[this.props.title];
    const iTitle =
      this.props.title == 'huggingface' && this.state.isMobile
        ? ' hf '
        : this.props.title;
    return (
      <>
        <a className={aClass} href={this.props.url} target="_blank">
          <FaIcon size="2em" />
          <span className={sClass} style={{ fontFamily: 'Poppins' }}>
            {iTitle}
          </span>
        </a>
      </>
    );
  }
}

export default class Header extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isMobile: window.innerWidth < 1000,
    };
    this.handleResize = this.handleResize.bind(this);
  }
  componentDidMount() {
    window.addEventListener('resize', this.handleResize);
    this.handleResize();
  }
  componentWillUnmount() {
    window.removeEventListener('resize', this.handleResize);
  }
  handleResize() {
    this.setState({ isMobile: window.innerWidth < 960 });
  }

  render() {
    const titleClass = `uk-${
      this.props.title.length > 15 ? 'h2' : 'h1'
    } uk-text-primary`;
    const baseStyle = this.props.header?.bg_image
      ? {
          backgroundImage: `url(${this.props.header.bg_image})`,
          backgroundSize: 'contain',
          backgroundRepeat: 'no-repeat',
          backgroundPosition: 'left',
          backgroundColor: '#030706',
        }
      : null;
    const backgroundStyle = this.state.isMobile
      ? null
      : {
          backgroundImage: `url(${this.props.header.bg_curve})`,
          backgroundSize: 'contain',
          backgroundRepeat: 'no-repeat',
          backgroundPosition: 'right',
          margin: '20px 10px 20px 0px',
        };
    return (
      <>
        <div
          className="uk-cover-container uk-background-secondary"
          style={baseStyle}
        >
          <div style={backgroundStyle}>
            <div className="uk-container uk-container-small uk-section">
              <div className="uk-text-center uk-text-bold">
                <p className={titleClass} style={{ fontSize: '1.9rem' }}>
                  {this.props.title}
                </p>
                <span
                  className="uk-label uk-label-primary uk-text-center uk-margin-small-bottom"
                  style={{ fontFamily: 'Poppins' }}
                >
                  {this.props.conference}
                </span>
              </div>
              <Authors
                authors={this.props.authors}
                affiliations={this.props.affiliations}
                meta={this.props.meta}
              />
              <div className="uk-text-center uk-margin-top">
                <a href="https://www.omron.com/sinicx" target="_blank">
                  <CorporateLogo
                    size="lg"
                    inverted={this.props.theme == 'dark' ? true : false}
                  />
                </a>
              </div>
              <div className="uk-flex uk-flex-center uk-margin-top">
                {Object.keys(this.props.resources).map((key) => (
                  <ResourceBtn
                    url={this.props.resources[key]}
                    title={key}
                    key={'header-' + key}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </>
    );
  }
}
