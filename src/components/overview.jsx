import React from 'react';
import { render } from 'react-dom';

import { marked } from 'marked';
import markedKatex from 'marked-katex-extension';
marked.use(markedKatex({ throwOnError: false }));

export default class Overview extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    return (
      <div className="uk-section">
        <img
          src={`${this.props.teaser}`}
          className="uk-align-center uk-responsive-width"
          alt=""
          width="90%"
        />
        {this.props.description && (
          <p className="uk-text-secondary uk-text-center uk-margin-bottom">
            <span
              className="uk-label uk-label-primary uk-text-center uk-text-bold"
              style={{ fontFamily: 'Poppins' }}
            >
              TL;DR
            </span>{' '}
            <span className="uk-text-secondary">{this.props.description}</span>
          </p>
        )}
        <h2 className="uk-heading-line uk-text-center">Overview</h2>
        <div
          dangerouslySetInnerHTML={{
            __html: marked.parse(this.props.overview),
          }}
        />
      </div>
    );
  }
}
